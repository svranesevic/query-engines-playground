use std::rc::Rc;

use arrow::datatypes::{DataType, Schema};

use query_core::{data_source::DataSource, logical_exprs::binary::Operator};
use volcano_engine::{
    physical_exprs::{aggregate::AggregateExpr, literal::Literal, PhysicalExpr},
    physical_plans::PhysicalPlan,
};

use super::CompileError;

#[derive(Clone)]
/// Scan descriptor extracted from the physical plan root.
///
/// This is the data-access boundary for JIT execution. It captures the input
/// source and projected scan columns before any fused filter/projection logic
/// is applied by generated code.
pub struct InputScan {
    pub data_source: Rc<dyn DataSource>,
    pub projection: Vec<String>,
    pub schema: Schema,
}

#[derive(Clone)]
/// Result of lowering a physical plan into JIT-friendly stages.
///
/// `program` contains the fused row-wise stage compiled by Cranelift.
/// `aggregate` stays as a separate post-kernel stage executed by runtime code.
pub struct LoweredPlan {
    pub input: InputScan,
    pub program: FusedProgram,
    pub aggregate: Option<AggregateStage>,
}

#[derive(Clone)]
/// Optional aggregate stage that runs after the fused kernel emits rows.
///
/// This remains outside the generated kernel so the JIT stage stays focused on
/// per-row filter/projection while aggregation is handled as a batch operator.
pub struct AggregateStage {
    pub group_expr: Vec<PhysicalExpr>,
    pub aggregate_expr: Vec<AggregateExpr>,
    pub schema: Schema,
}

#[derive(Clone)]
/// Typed fused program consumed by JIT codegen.
///
/// It is the key simplification boundary between planner expressions and native
/// code generation:
/// - `filter` is an optional Boolean predicate.
/// - `projection` are typed per-output expressions.
/// - `output_schema` defines emitted columns.
/// - `string_literals` is an intern table referenced by literal indices.
pub struct FusedProgram {
    pub filter: Option<FusedExpr>,
    pub projection: Vec<FusedExpr>,
    pub output_schema: Schema,
    pub string_literals: Vec<String>,
}

#[derive(Clone, Debug)]
/// Typed lowered expression node for fused kernel codegen.
pub enum FusedExpr {
    Column(FusedColumnExpr),
    Literal(FusedLiteral),
    Binary(FusedBinaryExpr),
}

#[derive(Clone, Debug)]
/// Column reference with resolved input index and data type.
///
/// Lowering resolves this once so codegen can emit direct loads without
/// repeatedly consulting schemas.
pub struct FusedColumnExpr {
    pub index: usize,
    pub data_type: DataType,
}

#[derive(Clone, Debug)]
pub enum FusedLiteral {
    Long(i64),
    UInt64(u64),
    Double(f64),
    Str { literal_idx: usize },
}

#[derive(Clone, Debug)]
/// Typed binary expression node with explicit output type.
///
/// Both children are already lowered/validated, so codegen only needs to map
/// operation + operand kinds to machine instructions or runtime helper calls.
pub struct FusedBinaryExpr {
    pub left: Box<FusedExpr>,
    pub right: Box<FusedExpr>,
    pub op: Operator,
    pub data_type: DataType,
}

impl FusedExpr {
    pub fn as_column_index(&self) -> Option<usize> {
        match self {
            FusedExpr::Column(col) => Some(col.index),
            _ => None,
        }
    }

    pub fn data_type(&self) -> &DataType {
        match self {
            FusedExpr::Column(col) => &col.data_type,
            FusedExpr::Literal(lit) => lit.data_type(),
            FusedExpr::Binary(binary) => &binary.data_type,
        }
    }
}

impl FusedLiteral {
    pub fn data_type(&self) -> &DataType {
        match self {
            FusedLiteral::Long(_) => &DataType::Int64,
            FusedLiteral::UInt64(_) => &DataType::UInt64,
            FusedLiteral::Double(_) => &DataType::Float64,
            FusedLiteral::Str { .. } => &DataType::Utf8,
        }
    }
}

pub fn lower_plan(plan: &PhysicalPlan) -> Result<LoweredPlan, CompileError> {
    let mut filter = None;
    let mut projection = None;
    let mut aggregate = None;

    let scan = collect_nodes(plan, &mut filter, &mut projection, &mut aggregate)?;

    let input_schema = scan.schema.clone();
    let (projection_exprs, output_schema) = if let Some((exprs, schema)) = projection {
        (exprs, schema)
    } else {
        (
            (0..input_schema.fields().len())
                .map(PhysicalExpr::column)
                .collect::<Vec<_>>(),
            input_schema.clone(),
        )
    };

    for (i, field) in output_schema.fields().iter().enumerate() {
        ensure_supported_type(field.data_type(), &format!("projection output column[{i}]"))?;
    }

    let mut string_literals = Vec::new();

    let lowered_filter = if let Some(predicate) = &filter {
        let expr = lower_expr(predicate, &input_schema, &mut string_literals, "filter")?;
        if expr.data_type() != &DataType::Boolean {
            return Err(CompileError::UnsupportedExpr(format!(
                "filter predicate must be Boolean, got {:?}",
                expr.data_type()
            )));
        }
        Some(expr)
    } else {
        None
    };

    let mut lowered_projection = Vec::with_capacity(projection_exprs.len());
    for (i, expr) in projection_exprs.iter().enumerate() {
        let lowered = lower_expr(
            expr,
            &input_schema,
            &mut string_literals,
            &format!("projection[{i}]"),
        )?;
        let expected = output_schema.field(i).data_type();
        if lowered.data_type() != expected {
            return Err(CompileError::UnsupportedExpr(format!(
                "projection[{i}] type mismatch: expression={:?}, schema={expected:?}",
                lowered.data_type(),
            )));
        }
        lowered_projection.push(lowered);
    }

    if let Some(agg) = &aggregate {
        for (i, field) in agg.schema.fields().iter().enumerate() {
            ensure_supported_type(field.data_type(), &format!("aggregate output column[{i}]"))?;
        }
    }

    Ok(LoweredPlan {
        input: scan,
        program: FusedProgram {
            filter: lowered_filter,
            projection: lowered_projection,
            output_schema,
            string_literals,
        },
        aggregate,
    })
}

fn collect_nodes(
    plan: &PhysicalPlan,
    filter: &mut Option<PhysicalExpr>,
    projection: &mut Option<(Vec<PhysicalExpr>, Schema)>,
    aggregate: &mut Option<AggregateStage>,
) -> Result<InputScan, CompileError> {
    match plan {
        PhysicalPlan::Scan(scan) => Ok(InputScan {
            data_source: scan.data_source(),
            projection: scan.projection().to_vec(),
            schema: scan.schema(),
        }),
        PhysicalPlan::Filter(f) => {
            if filter.is_some() {
                return Err(CompileError::UnsupportedPlan(
                    "multiple Filter nodes are not supported in fused JIT".to_string(),
                ));
            }
            *filter = Some(f.predicate().clone());
            collect_nodes(f.input(), filter, projection, aggregate)
        }
        PhysicalPlan::Projection(p) => {
            if projection.is_some() {
                return Err(CompileError::UnsupportedPlan(
                    "multiple Projection nodes are not supported in fused JIT".to_string(),
                ));
            }
            *projection = Some((p.exprs().to_vec(), p.schema()));
            collect_nodes(p.input(), filter, projection, aggregate)
        }
        PhysicalPlan::Aggregate(a) => {
            if aggregate.is_some() {
                return Err(CompileError::UnsupportedPlan(
                    "multiple Aggregate nodes are not supported".to_string(),
                ));
            }
            *aggregate = Some(AggregateStage {
                group_expr: a.group_expr().to_vec(),
                aggregate_expr: a.aggregate_expr().to_vec(),
                schema: a.schema(),
            });
            collect_nodes(a.input(), filter, projection, aggregate)
        }
    }
}

fn lower_expr(
    expr: &PhysicalExpr,
    schema: &Schema,
    string_literals: &mut Vec<String>,
    location: &str,
) -> Result<FusedExpr, CompileError> {
    match expr {
        PhysicalExpr::Column(col) => {
            if col.i >= schema.fields().len() {
                return Err(CompileError::UnsupportedExpr(format!(
                    "{location}: column index {} out of bounds for schema with {} fields",
                    col.i,
                    schema.fields().len()
                )));
            }
            let data_type = schema.field(col.i).data_type().clone();
            ensure_supported_type(&data_type, location)?;
            Ok(FusedExpr::Column(FusedColumnExpr {
                index: col.i,
                data_type,
            }))
        }
        PhysicalExpr::Literal(lit) => {
            let lowered = match lit {
                Literal::Long(v) => FusedLiteral::Long(*v),
                Literal::UInt64(v) => FusedLiteral::UInt64(*v),
                Literal::Double(v) => FusedLiteral::Double(*v),
                Literal::Str(v) => FusedLiteral::Str {
                    literal_idx: intern_string_literal(v, string_literals),
                },
            };
            ensure_supported_type(lowered.data_type(), location)?;
            Ok(FusedExpr::Literal(lowered))
        }
        PhysicalExpr::Binary(binary) => {
            let left = lower_expr(binary.left(), schema, string_literals, location)?;
            let right = lower_expr(binary.right(), schema, string_literals, location)?;
            let left_type = left.data_type();
            let right_type = right.data_type();

            if !is_binary_compatible(left_type, right_type) {
                return Err(CompileError::UnsupportedExpr(format!(
                    "{location}: binary type mismatch left={left_type:?}, right={right_type:?}"
                )));
            }

            let data_type = match binary.op() {
                Operator::And | Operator::Or => {
                    if left_type != &DataType::Boolean {
                        return Err(CompileError::UnsupportedExpr(format!(
                            "{location}: logical operator {:?} requires Boolean operands, got {left_type:?}",
                            binary.op()
                        )));
                    }
                    DataType::Boolean
                }
                Operator::Gt
                | Operator::GtEq
                | Operator::Lt
                | Operator::LtEq
                | Operator::Eq
                | Operator::NotEq => DataType::Boolean,
            };

            Ok(FusedExpr::Binary(FusedBinaryExpr {
                left: Box::new(left),
                right: Box::new(right),
                op: binary.op().clone(),
                data_type,
            }))
        }
        PhysicalExpr::Aggregate(_) => Err(CompileError::UnsupportedExpr(format!(
            "{location}: aggregate expressions are not supported in fused JIT"
        ))),
    }
}

fn ensure_supported_type(data_type: &DataType, location: &str) -> Result<(), CompileError> {
    match data_type {
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Boolean
        | DataType::Utf8
        | DataType::Utf8View => Ok(()),
        other => Err(CompileError::UnsupportedType(format!(
            "{location}: unsupported type {other:?}"
        ))),
    }
}

fn is_binary_compatible(left: &DataType, right: &DataType) -> bool {
    if left == right {
        return true;
    }
    matches!(
        (left, right),
        (DataType::Utf8, DataType::Utf8View) | (DataType::Utf8View, DataType::Utf8)
    )
}

fn intern_string_literal(value: &str, out: &mut Vec<String>) -> usize {
    if let Some(pos) = out.iter().position(|s| s == value) {
        pos
    } else {
        out.push(value.to_string());
        out.len() - 1
    }
}

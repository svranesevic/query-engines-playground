mod codegen;
mod lowering;
mod runtime;

use std::sync::Arc;
use std::{fs, path::Path};

use arrow::array::RecordBatch;

use volcano_engine::physical_plans::PhysicalPlan;

use self::{codegen::CraneliftFusedKernel, lowering::LoweredPlan};

#[derive(Debug)]
pub enum CompileError {
    UnsupportedPlan(String),
    UnsupportedExpr(String),
    UnsupportedType(String),
    Internal(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::UnsupportedPlan(msg) => write!(f, "unsupported plan: {msg}"),
            CompileError::UnsupportedExpr(msg) => write!(f, "unsupported expression: {msg}"),
            CompileError::UnsupportedType(msg) => write!(f, "unsupported type: {msg}"),
            CompileError::Internal(msg) => write!(f, "internal compiler error: {msg}"),
        }
    }
}

impl std::error::Error for CompileError {}

pub enum CompiledPlan {
    Cranelift(CraneliftCompiledPlan),
}

impl CompiledPlan {
    pub fn execute(&mut self) -> Vec<RecordBatch> {
        match self {
            CompiledPlan::Cranelift(plan) => plan.execute(),
        }
    }

    pub fn clif_ir(&self) -> &str {
        match self {
            CompiledPlan::Cranelift(plan) => &plan.kernel.clif_ir,
        }
    }

    pub fn dump_clif_ir<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        fs::write(path, self.clif_ir())
    }
}

/// Executable JIT plan instance bound to concrete input batches.
///
/// Execution flow:
/// 1. Evaluate the fused kernel (`filter + projection`) per input batch.
/// 2. Optionally run a hash aggregate stage on the kernel output.
///
/// This struct is the runtime bridge between compiled machine code and Arrow
/// record batches.
pub struct CraneliftCompiledPlan {
    lowered: LoweredPlan,
    kernel: CraneliftFusedKernel,
    input_batches: Vec<RecordBatch>,
}

impl CraneliftCompiledPlan {
    fn execute(&mut self) -> Vec<RecordBatch> {
        let batches = if self.lowered.program.filter.is_none()
            && self
                .lowered
                .program
                .projection
                .iter()
                .all(|expr| expr.as_column_index().is_some())
        {
            let schema = Arc::new(self.lowered.program.output_schema.clone());
            self.input_batches
                .iter()
                .map(|batch| {
                    let cols = self
                        .lowered
                        .program
                        .projection
                        .iter()
                        .map(|expr| batch.column(expr.as_column_index().unwrap()).clone())
                        .collect::<Vec<_>>();
                    RecordBatch::try_new(schema.clone(), cols).unwrap()
                })
                .collect::<Vec<_>>()
        } else {
            self.input_batches
                .iter()
                .map(|batch| {
                    runtime::execute_batch(self.kernel.func, batch, &self.lowered.program).unwrap()
                })
                .collect::<Vec<_>>()
        };

        if let Some(agg) = &self.lowered.aggregate {
            runtime::execute_hash_aggregate(
                &batches,
                &agg.group_expr,
                &agg.aggregate_expr,
                &agg.schema,
            )
        } else {
            batches
        }
    }
}

pub fn compile(plan: Arc<PhysicalPlan>) -> Result<CompiledPlan, CompileError> {
    let lowered = lowering::lower_plan(plan.as_ref())?;
    let kernel = codegen::compile_fused_kernel(&lowered.program)?;
    if std::env::var("JIT_DUMP_CLIF_STDOUT").is_ok() {
        eprintln!(
            "=== JIT CLIF IR ===\n{}\n=== /JIT CLIF IR ===",
            kernel.clif_ir
        );
    }
    let input_batches = lowered
        .input
        .data_source
        .scan(lowered.input.projection.clone());
    Ok(CompiledPlan::Cranelift(CraneliftCompiledPlan {
        lowered,
        kernel,
        input_batches,
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use query_core::{
        data_frame::ExecutionContext,
        logical_exprs::{aggregate::AggregateExpr, LogicalExpr},
        optimizer::optimize,
    };
    use volcano_engine::planner::create_physical_plan;

    use super::compile;

    #[test]
    fn jit_filter_projection_matches_interpreter() {
        let age_col = LogicalExpr::col("age");

        let logical_plan = ExecutionContext::csv("employee.csv")
            .filter(age_col.clone().gte(LogicalExpr::lit_long(20)))
            .project(vec![age_col])
            .logical_plan();

        let logical_plan = optimize(logical_plan);
        let mut interpreted = create_physical_plan(&logical_plan);
        let interpreted_out = interpreted.execute();

        let physical = create_physical_plan(&logical_plan);
        let mut compiled = compile(Arc::new(physical)).unwrap();
        let compiled_out = compiled.execute();

        assert_eq!(interpreted_out, compiled_out);
    }

    #[test]
    fn jit_scan_only_matches_interpreter() {
        let logical_plan = ExecutionContext::csv("employee.csv")
            .project(vec![LogicalExpr::col("age")])
            .logical_plan();
        let logical_plan = optimize(logical_plan);

        let mut interpreted = create_physical_plan(&logical_plan);
        let interpreted_out = interpreted.execute();

        let physical = create_physical_plan(&logical_plan);
        let mut compiled = compile(Arc::new(physical)).unwrap();
        let compiled_out = compiled.execute();

        assert_eq!(interpreted_out, compiled_out);
    }

    #[test]
    fn jit_string_filter_projection_matches_interpreter() {
        let first_name_col = LogicalExpr::col("first_name");
        let last_name_col = LogicalExpr::col("last_name");
        let age_col = LogicalExpr::col("age");

        let logical_plan = ExecutionContext::csv("employee.csv")
            .filter(first_name_col.clone().neq(last_name_col))
            .project(vec![first_name_col, age_col])
            .logical_plan();

        let mut interpreted = create_physical_plan(&logical_plan);
        let interpreted_out = interpreted.execute();

        let physical = create_physical_plan(&logical_plan);
        let mut compiled = compile(Arc::new(physical)).unwrap();
        let compiled_out = compiled.execute();

        assert_eq!(interpreted_out, compiled_out);
    }

    #[test]
    fn jit_string_literal_projection_matches_interpreter() {
        let age_col = LogicalExpr::col("age");
        let logical_plan = ExecutionContext::csv("employee.csv")
            .project(vec![LogicalExpr::lit_str("x"), age_col])
            .logical_plan();

        let mut interpreted = create_physical_plan(&logical_plan);
        let interpreted_out = interpreted.execute();

        let physical = create_physical_plan(&logical_plan);
        let mut compiled = compile(Arc::new(physical)).unwrap();
        let compiled_out = compiled.execute();

        assert_eq!(interpreted_out, compiled_out);
    }

    #[test]
    fn jit_operator_matrix_matches_interpreter() {
        let age = LogicalExpr::col("age");
        let first = LogicalExpr::col("first_name");
        let last = LogicalExpr::col("last_name");

        let scenarios = vec![
            ExecutionContext::csv("employee.csv")
                .filter(age.clone().gt(LogicalExpr::lit_long(20)))
                .project(vec![age.clone()])
                .logical_plan(),
            ExecutionContext::csv("employee.csv")
                .filter(age.clone().gte(LogicalExpr::lit_long(42)))
                .project(vec![age.clone()])
                .logical_plan(),
            ExecutionContext::csv("employee.csv")
                .filter(age.clone().lt(LogicalExpr::lit_long(30)))
                .project(vec![age.clone()])
                .logical_plan(),
            ExecutionContext::csv("employee.csv")
                .filter(age.clone().lteq(LogicalExpr::lit_long(30)))
                .project(vec![age.clone()])
                .logical_plan(),
            ExecutionContext::csv("employee.csv")
                .filter(age.clone().eq(LogicalExpr::lit_long(42)))
                .project(vec![age.clone()])
                .logical_plan(),
            ExecutionContext::csv("employee.csv")
                .filter(age.clone().neq(LogicalExpr::lit_long(42)))
                .project(vec![age.clone()])
                .logical_plan(),
            ExecutionContext::csv("employee.csv")
                .filter(first.clone().lt(last.clone()))
                .project(vec![first.clone(), last.clone()])
                .logical_plan(),
            ExecutionContext::csv("employee.csv")
                .filter(first.clone().lteq(last.clone()))
                .project(vec![first.clone(), last.clone()])
                .logical_plan(),
            ExecutionContext::csv("employee.csv")
                .filter(first.clone().gt(last.clone()))
                .project(vec![first.clone(), last.clone()])
                .logical_plan(),
            ExecutionContext::csv("employee.csv")
                .filter(first.clone().gte(last.clone()))
                .project(vec![first.clone(), last.clone()])
                .logical_plan(),
            ExecutionContext::csv("employee.csv")
                .filter(
                    age.clone()
                        .gt(LogicalExpr::lit_long(20))
                        .and(age.clone().lt(LogicalExpr::lit_long(40))),
                )
                .project(vec![age.clone()])
                .logical_plan(),
            ExecutionContext::csv("employee.csv")
                .filter(
                    age.clone()
                        .lt(LogicalExpr::lit_long(20))
                        .or(age.clone().eq(LogicalExpr::lit_long(42))),
                )
                .project(vec![age.clone()])
                .logical_plan(),
        ];

        for logical in scenarios {
            let mut interpreted = create_physical_plan(&logical);
            let interpreted_out = interpreted.execute();

            let physical = create_physical_plan(&logical);
            let mut compiled = compile(Arc::new(physical)).unwrap();
            let compiled_out = compiled.execute();
            assert_eq!(interpreted_out, compiled_out);
        }
    }

    #[test]
    fn jit_aggregate_matches_interpreter() {
        let age_col = LogicalExpr::col("age");
        let first_name = LogicalExpr::col("first_name");
        let logical_plan = ExecutionContext::csv("employee.csv")
            .aggregate(
                vec![first_name],
                vec![
                    AggregateExpr::count(age_col.clone()),
                    AggregateExpr::avg(age_col),
                ],
            )
            .logical_plan();

        let mut interpreted = create_physical_plan(&logical_plan);
        let interpreted_out = interpreted.execute();

        let physical = create_physical_plan(&logical_plan);
        let mut compiled = compile(Arc::new(physical)).unwrap();
        let compiled_out = compiled.execute();
        assert_eq!(interpreted_out, compiled_out);
    }
}

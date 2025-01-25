use crate::{
    logical_exprs::{
        aggregate::AggregateExpr as LogicalAggregateExpr, binary::Operator as LogicalBinaryOp,
        literal::Literal as LogicalLiteral, LogicalExpr,
    },
    logical_plans::LogicalPlan,
    physical_exprs::{aggregate::AggregateExpr as PhysicalAggregateExpr, PhysicalExpr},
    physical_plans::PhysicalPlan,
};

pub fn create_physical_plan(plan: &LogicalPlan) -> PhysicalPlan {
    match plan {
        LogicalPlan::Scan(scan) => {
            PhysicalPlan::scan(scan.data_source.clone(), scan.projection.clone())
        }
        LogicalPlan::Projection(projection) => {
            let input = Box::new(create_physical_plan(projection.input.as_ref()));
            let schema = projection.schema();
            let exprs = projection
                .exprs
                .iter()
                .map(|e| create_physical_expr(e, plan))
                .collect();
            PhysicalPlan::projection(input, schema, exprs)
        }
        LogicalPlan::Filter(filter) => {
            let input = create_physical_plan(filter.input.as_ref());
            let expr = create_physical_expr(&filter.expr, plan);
            PhysicalPlan::filter(Box::new(input), expr)
        }
        LogicalPlan::Aggregate(aggregate) => {
            let input = create_physical_plan(aggregate.input.as_ref());
            let group_by: Vec<_> = aggregate
                .group_by
                .iter()
                .map(|e| create_physical_expr(e, plan))
                .collect();
            let aggr_expr: Vec<_> = aggregate
                .agg
                .iter()
                .map(|e| create_aggregate_expr(e, plan))
                .collect();
            PhysicalPlan::aggregate(Box::new(input), group_by, aggr_expr, aggregate.schema())
        }
    }
}

fn create_aggregate_expr(
    aggregate: &LogicalAggregateExpr,
    plan: &LogicalPlan,
) -> PhysicalAggregateExpr {
    let expr = Box::new(create_physical_expr(aggregate.input(), plan));
    match aggregate {
        LogicalAggregateExpr::Max(_) => PhysicalAggregateExpr::max(expr),
        LogicalAggregateExpr::Sum(_) => PhysicalAggregateExpr::sum(expr),
        LogicalAggregateExpr::Min(_) => PhysicalAggregateExpr::min(expr),
        LogicalAggregateExpr::Avg(_) => PhysicalAggregateExpr::avg(expr),
        LogicalAggregateExpr::Count(_) => PhysicalAggregateExpr::count(expr),
    }
}

fn create_physical_expr(expr: &LogicalExpr, plan: &LogicalPlan) -> PhysicalExpr {
    match expr {
        LogicalExpr::Column(col) => {
            let idx = plan.schema().index_of(&col.name).unwrap();
            PhysicalExpr::column(idx)
        }
        LogicalExpr::Literal(lit) => match lit {
            LogicalLiteral::Str(str) => PhysicalExpr::lit_str(str.clone()),
            LogicalLiteral::Long(i64) => PhysicalExpr::lit_long(*i64),
        },
        LogicalExpr::Binary(binary) => {
            let left = create_physical_expr(binary.left.as_ref(), plan);
            let right = create_physical_expr(binary.right.as_ref(), plan);

            match binary.op {
                LogicalBinaryOp::Gt => PhysicalExpr::gt(left, right),
                LogicalBinaryOp::GtEq => PhysicalExpr::gteq(left, right),
                LogicalBinaryOp::Lt => PhysicalExpr::lt(left, right),
                LogicalBinaryOp::LtEq => PhysicalExpr::lteq(left, right),
                LogicalBinaryOp::Eq => PhysicalExpr::eq(left, right),
                LogicalBinaryOp::NotEq => PhysicalExpr::neq(left, right),
                LogicalBinaryOp::And => PhysicalExpr::and(left, right),
                LogicalBinaryOp::Or => PhysicalExpr::or(left, right),
            }
        }
        LogicalExpr::Aggregate(aggregate) => {
            panic!("Unexpected aggregate expression: {}", aggregate)
        }
    }
}

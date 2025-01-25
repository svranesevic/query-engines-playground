use crate::{
    logical_exprs::binary::Operator as LogicalBinaryOp,
    logical_exprs::literal::Literal as LogicalLiteral, logical_exprs::LogicalExpr,
    logical_plans::LogicalPlan, physical_exprs::PhysicalExpr, physical_plans::PhysicalPlan,
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
            // let input = create_physical_plan(aggregate.input.as_ref());
            // let group_expr = aggregate
            //     .group_expr
            //     .iter()
            //     .map(|e| create_physical_expr(e, plan))
            //     .collect();
            // let aggr_expr = aggregate
            //     .aggr_expr
            //     .iter()
            //     .map(|e| create_physical_expr(e, plan))
            //     .collect();
            // PhysicalPlan::aggregate(Box::new(input), group_expr, aggr_expr)
            todo!()
        }
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
            // let input = create_physical_expr(aggregate.expr(), plan);
            // match aggregate {
            //     Aggregate::Sum(_) => PhysicalExpr::sum(input),
            //     Aggregate::Min(_) => PhysicalExpr::min(input),
            //     Aggregate::Max(_) => PhysicalExpr::max(input),
            //     Aggregate::Avg(_) => PhysicalExpr::avg(input),
            // }
            todo!()
        }
    }
}

use std::collections::HashSet;

use crate::{
    logical_exprs::{aggregate::AggregateExpr, LogicalExpr},
    logical_plans::LogicalPlan,
};

trait OptimizerRule {
    fn optimize(self, plan: LogicalPlan) -> LogicalPlan;
}

struct ProjectionPushDown {
    cols: Vec<String>,
    col_set: HashSet<String>,
}

impl ProjectionPushDown {
    fn new() -> Self {
        Self {
            cols: vec![],
            col_set: HashSet::new(),
        }
    }

    fn add_cols(&mut self, cols: Vec<String>) {
        for col in cols {
            if self.col_set.insert(col.clone()) {
                self.cols.push(col);
            }
        }
    }
}

impl OptimizerRule for ProjectionPushDown {
    // Collect referenced columns, and assign them to the scan node
    fn optimize(mut self, plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Scan(p) => LogicalPlan::scan(p.path, p.data_source, self.cols),
            LogicalPlan::Filter(p) => {
                let cols = cols_from_expr(&p.expr);
                self.add_cols(cols);
                let input = self.optimize(*p.input);
                LogicalPlan::filter(input, p.expr)
            }
            LogicalPlan::Projection(p) => {
                let cols = cols_from_exprs(&p.exprs);
                self.add_cols(cols);
                let input = self.optimize(*p.input);
                LogicalPlan::projection(input, p.exprs)
            }
            LogicalPlan::Aggregate(p) => {
                let mut cols = cols_from_exprs(&p.group_by);
                cols.extend(cols_from_aggregate_exprs(&p.agg));
                self.add_cols(cols);
                let input = self.optimize(*p.input);
                LogicalPlan::aggregate(input, p.group_by, p.agg)
            }
        }
    }
}

fn cols_from_exprs(exprs: &[LogicalExpr]) -> Vec<String> {
    exprs.iter().flat_map(cols_from_expr).collect()
}

fn cols_from_expr(expr: &LogicalExpr) -> Vec<String> {
    match expr {
        LogicalExpr::Column(c) => vec![c.name.clone()],
        LogicalExpr::Literal(_) => vec![],
        LogicalExpr::Binary(b) => {
            let mut left_columns = cols_from_expr(&b.left);
            let mut right_columns = cols_from_expr(&b.right);
            left_columns.append(&mut right_columns);
            left_columns
        }
        LogicalExpr::Aggregate(a) => cols_from_expr(a.input()),
    }
}

fn cols_from_aggregate_exprs(exprs: &[AggregateExpr]) -> Vec<String> {
    exprs
        .iter()
        .flat_map(|agg| cols_from_expr(agg.input()))
        .collect()
}

pub fn optimize(plan: LogicalPlan) -> LogicalPlan {
    ProjectionPushDown::new().optimize(plan)
}

#[cfg(test)]
mod tests {
    use crate::{
        data_frame::ExecutionContext, logical_exprs::aggregate::AggregateExpr,
        logical_exprs::LogicalExpr, logical_plans::LogicalPlan,
    };

    use super::optimize;

    #[test]
    fn scan_projection_order_is_deterministic_first_seen() {
        let first_name_col = LogicalExpr::col("first_name");
        let age_col = LogicalExpr::col("age");
        let plan = ExecutionContext::csv("employee.csv")
            .filter(age_col.gte(LogicalExpr::lit_long(20)))
            .project(vec![
                first_name_col,
                LogicalExpr::col("age"),
                LogicalExpr::col("age"),
            ])
            .logical_plan();

        let optimized = optimize(plan);
        let mut current = &optimized;
        let scan = loop {
            match current {
                LogicalPlan::Scan(scan) => break scan,
                LogicalPlan::Projection(p) => current = p.input.as_ref(),
                LogicalPlan::Filter(f) => current = f.input.as_ref(),
                LogicalPlan::Aggregate(a) => current = a.input.as_ref(),
            }
        };

        assert_eq!(
            scan.projection,
            vec!["first_name".to_string(), "age".to_string()]
        );
    }

    #[test]
    fn aggregate_pushdown_includes_aggregate_input_columns() {
        let plan = ExecutionContext::csv("employee.csv")
            .aggregate(vec![], vec![AggregateExpr::count(LogicalExpr::col("age"))])
            .logical_plan();

        let optimized = optimize(plan);
        let mut current = &optimized;
        let scan = loop {
            match current {
                LogicalPlan::Scan(scan) => break scan,
                LogicalPlan::Projection(p) => current = p.input.as_ref(),
                LogicalPlan::Filter(f) => current = f.input.as_ref(),
                LogicalPlan::Aggregate(a) => current = a.input.as_ref(),
            }
        };

        assert_eq!(scan.projection, vec!["age".to_string()]);
    }
}

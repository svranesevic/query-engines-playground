use std::collections::HashSet;

use crate::{logical_exprs::LogicalExpr, logical_plans::LogicalPlan};

trait OptimizerRule {
    fn optimize(self, plan: LogicalPlan) -> LogicalPlan;
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
    }
}

struct ProjectionPushDown {
    cols: HashSet<String>,
}

impl ProjectionPushDown {
    fn new() -> Self {
        Self {
            cols: HashSet::new(),
        }
    }
}

impl OptimizerRule for ProjectionPushDown {
    // Depth First traversal collecting columns, then assigning them to Scan when encountered
    fn optimize(mut self, plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Scan(p) => {
                LogicalPlan::scan(p.path, p.data_source, self.cols.into_iter().collect())
            }
            LogicalPlan::Filter(p) => {
                let cols = cols_from_expr(&p.expr);
                self.cols.extend(cols);
                let input = self.optimize(*p.input);
                LogicalPlan::filter(input, p.expr)
            }
            LogicalPlan::Projection(p) => {
                let cols = cols_from_exprs(&p.exprs);
                self.cols.extend(cols);
                let input = self.optimize(*p.input);
                LogicalPlan::projection(input, p.exprs)
            }
        }
    }
}

pub fn optimize(plan: LogicalPlan) -> LogicalPlan {
    ProjectionPushDown::new().optimize(plan)
}

use super::LogicalPlan;
use crate::LogicalExpr;
use arrow::datatypes::Schema;

pub struct Filter {
    pub input: Box<LogicalPlan>,
    pub expr: LogicalExpr,
}

impl Filter {
    pub fn new(input: LogicalPlan, expr: LogicalExpr) -> Self {
        Self {
            input: Box::new(input),
            expr,
        }
    }

    pub fn schema(&self) -> Schema {
        self.input.schema()
    }

    pub fn children(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }
}

impl std::fmt::Display for Filter {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Filter: {}", self.expr)
    }
}

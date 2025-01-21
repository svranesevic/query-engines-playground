use super::LogicalPlan;
use crate::LogicalExpr;
use arrow::datatypes::Schema;

pub struct Projection {
    pub input: Box<LogicalPlan>,
    pub exprs: Vec<LogicalExpr>,
}

impl Projection {
    pub fn new(input: Box<LogicalPlan>, exprs: Vec<LogicalExpr>) -> Self {
        Self { input, exprs }
    }

    pub fn schema(&self) -> Schema {
        let fields: Vec<_> = self.exprs.iter().map(|e| e.to_field(&self.input)).collect();
        Schema::new(fields)
    }

    pub fn children(&self) -> Vec<&LogicalPlan> {
        vec![&*self.input]
    }
}

impl std::fmt::Display for Projection {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let exprs = self.exprs.iter().map(|e| e.to_string()).collect::<Vec<_>>();
        write!(f, "Projection: {:?}", exprs)
    }
}

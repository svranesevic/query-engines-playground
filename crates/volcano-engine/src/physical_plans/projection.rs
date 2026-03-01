use std::sync::Arc;

use arrow::{array::RecordBatch, datatypes::Schema};

use super::PhysicalPlan;
use crate::physical_exprs::PhysicalExpr;

pub struct ProjectionExec {
    input: Box<PhysicalPlan>,
    schema: Schema,
    exprs: Vec<PhysicalExpr>,
}

impl ProjectionExec {
    pub fn new(input: Box<PhysicalPlan>, schema: Schema, exprs: Vec<PhysicalExpr>) -> Self {
        Self {
            input,
            schema,
            exprs,
        }
    }

    pub fn children(&self) -> Vec<&PhysicalPlan> {
        vec![&*self.input]
    }

    pub fn schema(&self) -> Schema {
        self.schema.clone()
    }

    pub fn execute(&mut self) -> Vec<RecordBatch> {
        let input = self.input.execute();
        input
            .into_iter()
            .map(|batch| {
                let cols = self.exprs.iter().map(|e| e.evaluate(&batch)).collect();
                RecordBatch::try_new(Arc::new(self.schema.clone()), cols).unwrap()
            })
            .collect()
    }

    pub fn input(&self) -> &PhysicalPlan {
        self.input.as_ref()
    }

    pub fn exprs(&self) -> &[PhysicalExpr] {
        &self.exprs
    }
}

impl std::fmt::Display for ProjectionExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let exprs = self.exprs.iter().map(|e| e.to_string()).collect::<Vec<_>>();
        write!(f, "ProjectionExec: {:?}", exprs)
    }
}

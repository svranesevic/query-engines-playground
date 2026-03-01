use arrow::{
    array::{BooleanArray, RecordBatch},
    compute::filter_record_batch,
    datatypes::Schema,
};

use crate::physical_exprs::PhysicalExpr;

use super::PhysicalPlan;

pub struct FilterExec {
    input: Box<PhysicalPlan>,
    predicate: PhysicalExpr,
}

impl FilterExec {
    pub fn new(input: Box<PhysicalPlan>, predicate: PhysicalExpr) -> Self {
        Self { input, predicate }
    }

    pub fn children(&self) -> Vec<&PhysicalPlan> {
        vec![&self.input]
    }

    pub fn schema(&self) -> Schema {
        self.input.schema()
    }

    pub fn execute(&mut self) -> Vec<RecordBatch> {
        let input = self.input.execute();

        input
            .into_iter()
            .map(|batch| {
                let predicate = self.predicate.evaluate(&batch);
                let predicate = predicate.as_any().downcast_ref::<BooleanArray>().unwrap();
                filter_record_batch(&batch, predicate).unwrap()
            })
            .collect()
    }

    pub fn input(&self) -> &PhysicalPlan {
        self.input.as_ref()
    }

    pub fn predicate(&self) -> &PhysicalExpr {
        &self.predicate
    }
}

impl std::fmt::Display for FilterExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FilterExec: predicate={}", self.predicate)
    }
}

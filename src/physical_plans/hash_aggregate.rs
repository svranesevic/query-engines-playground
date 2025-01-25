use arrow::{array::RecordBatch, datatypes::Schema};

use crate::physical_exprs::{aggregate::AggregateExpr, PhysicalExpr};

use super::PhysicalPlan;

pub struct HashAggregateExec {
    input: Box<PhysicalPlan>,
    group_expr: Vec<PhysicalExpr>,
    aggregate_expr: Vec<AggregateExpr>,
    schema: Schema,
}

impl HashAggregateExec {
    pub fn new(
        input: Box<PhysicalPlan>,
        group_expr: Vec<PhysicalExpr>,
        aggregate_expr: Vec<AggregateExpr>,
        schema: Schema,
    ) -> Self {
        Self {
            input,
            group_expr,
            aggregate_expr,
            schema,
        }
    }

    pub fn schema(&self) -> Schema {
        self.schema.clone()
    }

    pub fn children(&self) -> Vec<&PhysicalPlan> {
        vec![&self.input]
    }

    pub fn execute(&mut self) -> Vec<RecordBatch> {
        todo!()
    }
}

impl std::fmt::Display for HashAggregateExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let group_expr = self
            .group_expr
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>();
        let aggregate_expr = self
            .aggregate_expr
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>();
        write!(
            f,
            "HashAggregateExec: groupBy=[{:?}], aggr=[{:?}]",
            group_expr, aggregate_expr
        )
    }
}

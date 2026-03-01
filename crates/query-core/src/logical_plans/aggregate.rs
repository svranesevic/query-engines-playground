use super::LogicalPlan;
use crate::logical_exprs::{aggregate::AggregateExpr, LogicalExpr};
use arrow::datatypes::Schema;

pub struct Aggregate {
    pub input: Box<LogicalPlan>,
    pub group_by: Vec<LogicalExpr>,
    pub agg: Vec<AggregateExpr>,
}

impl Aggregate {
    pub fn new(
        input: Box<LogicalPlan>,
        group_expr: Vec<LogicalExpr>,
        aggr_expr: Vec<AggregateExpr>,
    ) -> Self {
        Self {
            input,
            group_by: group_expr,
            agg: aggr_expr,
        }
    }

    pub fn schema(&self) -> Schema {
        let mut group_fields: Vec<_> = self
            .group_by
            .iter()
            .map(|e| e.to_field(self.input.as_ref()))
            .collect();

        let mut agg_fields: Vec<_> = self
            .agg
            .iter()
            .map(|e| e.to_field(self.input.as_ref()))
            .collect();

        let mut fields = Vec::with_capacity(group_fields.len() + agg_fields.len());
        fields.append(&mut group_fields);
        fields.append(&mut agg_fields);

        Schema::new(fields)
    }

    pub fn children(&self) -> Vec<&LogicalPlan> {
        vec![&*self.input]
    }
}

impl std::fmt::Display for Aggregate {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let group_expr = self
            .group_by
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>();
        let aggr_expr = self.agg.iter().map(|e| e.to_string()).collect::<Vec<_>>();
        write!(
            f,
            "Aggregate: groupBy=[{:?}], aggr=[{:?}]",
            group_expr, aggr_expr
        )
    }
}

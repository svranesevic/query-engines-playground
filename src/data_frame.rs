use crate::{
    data_source::CsvDataSource,
    logical_exprs::{aggregate::AggregateExpr, LogicalExpr},
    logical_plans::LogicalPlan,
};
use arrow::datatypes::Schema;
use std::rc::Rc;

pub struct DataFrame {
    plan: LogicalPlan,
}

impl DataFrame {
    pub fn logical_plan(self) -> LogicalPlan {
        self.plan
    }

    pub fn project(self, expr: Vec<LogicalExpr>) -> Self {
        let plan = LogicalPlan::projection(self.plan, expr);
        DataFrame::new(plan)
    }

    pub fn filter(self, expr: LogicalExpr) -> Self {
        let plan = LogicalPlan::filter(self.plan, expr);
        DataFrame::new(plan)
    }

    pub fn aggregate(self, group_by: Vec<LogicalExpr>, agg: Vec<AggregateExpr>) -> Self {
        let plan = LogicalPlan::aggregate(self.plan, group_by, agg);
        DataFrame::new(plan)
    }

    fn new(plan: LogicalPlan) -> Self {
        Self { plan }
    }

    fn schema(&self) -> Schema {
        self.plan.schema()
    }
}

pub struct ExecutionContext;
impl ExecutionContext {
    pub fn csv(filename: &str) -> DataFrame {
        let data_source = CsvDataSource::new(filename.to_string());
        let plan = LogicalPlan::scan(filename.to_string(), Rc::new(data_source), vec![]);
        DataFrame::new(plan)
    }
}

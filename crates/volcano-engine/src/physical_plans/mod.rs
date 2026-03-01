pub mod filter;
pub mod hash_aggregate;
pub mod projection;
pub mod scan;

use std::rc::Rc;

use arrow::{array::RecordBatch, datatypes::Schema};
use filter::FilterExec;
use hash_aggregate::HashAggregateExec;
use projection::ProjectionExec;
use scan::ScanExec;

use crate::physical_exprs::{aggregate::AggregateExpr, PhysicalExpr};
use query_core::data_source::DataSource;

pub enum PhysicalPlan {
    Scan(ScanExec),
    Projection(ProjectionExec),
    Filter(FilterExec),
    Aggregate(HashAggregateExec),
}

impl PhysicalPlan {
    pub fn scan(data_source: Rc<dyn DataSource>, projection: Vec<String>) -> Self {
        PhysicalPlan::Scan(ScanExec::new(data_source, projection))
    }

    pub fn projection(input: Box<PhysicalPlan>, schema: Schema, exprs: Vec<PhysicalExpr>) -> Self {
        PhysicalPlan::Projection(ProjectionExec::new(input, schema, exprs))
    }

    pub fn filter(input: Box<PhysicalPlan>, predicate: PhysicalExpr) -> Self {
        PhysicalPlan::Filter(FilterExec::new(input, predicate))
    }

    pub fn aggregate(
        input: Box<PhysicalPlan>,
        group_expr: Vec<PhysicalExpr>,
        aggr_expr: Vec<AggregateExpr>,
        schema: Schema,
    ) -> Self {
        PhysicalPlan::Aggregate(HashAggregateExec::new(input, group_expr, aggr_expr, schema))
    }

    pub fn children(&self) -> Vec<&PhysicalPlan> {
        match self {
            PhysicalPlan::Scan(scan) => scan.children(),
            PhysicalPlan::Projection(proj) => proj.children(),
            PhysicalPlan::Filter(filter) => filter.children(),
            PhysicalPlan::Aggregate(agg) => agg.children(),
        }
    }

    pub fn schema(&self) -> Schema {
        match self {
            PhysicalPlan::Scan(scan) => scan.schema(),
            PhysicalPlan::Projection(proj) => proj.schema(),
            PhysicalPlan::Filter(filter) => filter.schema(),
            PhysicalPlan::Aggregate(agg) => agg.schema(),
        }
    }

    // TODO(Sava): async, stream?
    pub fn execute(&mut self) -> Vec<RecordBatch> {
        match self {
            PhysicalPlan::Scan(scan) => scan.execute(),
            PhysicalPlan::Projection(proj) => proj.execute(),
            PhysicalPlan::Filter(filter) => filter.execute(),
            PhysicalPlan::Aggregate(agg) => agg.execute(),
        }
    }

    pub fn format(&self) -> String {
        self.format_with_indent(0)
    }

    fn format_with_indent(&self, indent: usize) -> String {
        let mut sb = String::new();

        (0..indent).for_each(|_| sb.push_str("  "));
        sb.push_str(&self.to_string());

        for c in self.children() {
            sb.push('\n');
            sb.push_str(&c.format_with_indent(indent + 1));
        }

        sb
    }
}

impl std::fmt::Display for PhysicalPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhysicalPlan::Scan(p) => p.fmt(f),
            PhysicalPlan::Projection(p) => p.fmt(f),
            PhysicalPlan::Filter(p) => p.fmt(f),
            PhysicalPlan::Aggregate(p) => p.fmt(f),
        }
    }
}

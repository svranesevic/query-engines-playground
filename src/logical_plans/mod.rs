pub mod aggregate;
pub mod filter;
pub mod projection;
pub mod scan;

use std::rc::Rc;

use crate::data_source::DataSource;
use crate::logical_exprs::aggregate::AggregateExpr;
use crate::logical_exprs::LogicalExpr;
use aggregate::Aggregate;
use arrow::datatypes::Schema;
use filter::Filter;
use projection::Projection;
use scan::Scan;

pub enum LogicalPlan {
    Scan(Scan),
    Projection(Projection),
    Filter(Filter),
    Aggregate(Aggregate),
}

impl LogicalPlan {
    pub fn scan(
        path: String,
        data_source: Rc<dyn DataSource>,
        projection: Vec<String>,
    ) -> LogicalPlan {
        LogicalPlan::Scan(Scan::new(path, data_source, projection))
    }

    pub fn projection(input: LogicalPlan, exprs: Vec<LogicalExpr>) -> LogicalPlan {
        LogicalPlan::Projection(Projection::new(Box::new(input), exprs))
    }

    pub fn filter(input: LogicalPlan, expr: LogicalExpr) -> LogicalPlan {
        LogicalPlan::Filter(Filter::new(input, expr))
    }

    pub fn aggregate(
        input: LogicalPlan,
        group_by: Vec<LogicalExpr>,
        agg: Vec<AggregateExpr>,
    ) -> LogicalPlan {
        LogicalPlan::Aggregate(Aggregate::new(Box::new(input), group_by, agg))
    }

    pub fn schema(&self) -> Schema {
        match self {
            LogicalPlan::Scan(s) => s.schema(),
            LogicalPlan::Projection(p) => p.schema(),
            LogicalPlan::Filter(f) => f.schema(),
            LogicalPlan::Aggregate(a) => a.schema(),
        }
    }
    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::Scan(s) => s.children(),
            LogicalPlan::Projection(p) => p.children(),
            LogicalPlan::Filter(f) => f.children(),
            LogicalPlan::Aggregate(a) => a.children(),
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

impl std::fmt::Display for LogicalPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LogicalPlan::Scan(p) => p.fmt(f),
            LogicalPlan::Projection(p) => p.fmt(f),
            LogicalPlan::Filter(p) => p.fmt(f),
            LogicalPlan::Aggregate(p) => p.fmt(f),
        }
    }
}

use std::rc::Rc;

use arrow::datatypes::Field;

use crate::logical_plans::LogicalPlan;

use super::LogicalExpr;

#[derive(Clone)]
pub enum AggregateExpr {
    Sum(Rc<LogicalExpr>),
    Min(Rc<LogicalExpr>),
    Max(Rc<LogicalExpr>),
    Avg(Rc<LogicalExpr>),
    Count(Rc<Count>),
}

impl AggregateExpr {
    pub fn sum(expr: LogicalExpr) -> Self {
        AggregateExpr::Sum(Rc::new(expr))
    }

    pub fn min(expr: LogicalExpr) -> Self {
        AggregateExpr::Min(Rc::new(expr))
    }

    pub fn max(expr: LogicalExpr) -> Self {
        AggregateExpr::Max(Rc::new(expr))
    }

    pub fn avg(expr: LogicalExpr) -> Self {
        AggregateExpr::Avg(Rc::new(expr))
    }

    pub fn count(expr: LogicalExpr) -> Self {
        AggregateExpr::Count(Rc::new(Count::new(expr)))
    }

    pub fn input(&self) -> &LogicalExpr {
        match self {
            AggregateExpr::Sum(input) => input,
            AggregateExpr::Min(input) => input,
            AggregateExpr::Max(input) => input,
            AggregateExpr::Avg(input) => input,
            AggregateExpr::Count(agg) => agg.input.as_ref(),
        }
    }

    pub fn to_field(&self, input: &LogicalPlan) -> Field {
        match self {
            AggregateExpr::Sum(expr) => expr.to_field(input),
            AggregateExpr::Min(expr) => expr.to_field(input),
            AggregateExpr::Max(expr) => expr.to_field(input),
            AggregateExpr::Avg(expr) => expr
                .to_field(input)
                .with_data_type(arrow::datatypes::DataType::Float64),
            AggregateExpr::Count(agg) => agg.to_field(input),
        }
    }
}

impl std::fmt::Display for AggregateExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AggregateExpr::Sum(expr) => write!(f, "SUM({})", expr),
            AggregateExpr::Min(expr) => write!(f, "MIN({})", expr),
            AggregateExpr::Max(expr) => write!(f, "MAX({})", expr),
            AggregateExpr::Avg(expr) => write!(f, "AVG({})", expr),
            AggregateExpr::Count(agg) => agg.fmt(f),
        }
    }
}

#[derive(Clone)]
pub struct Count {
    input: Rc<LogicalExpr>,
}

impl Count {
    fn new(input: LogicalExpr) -> Self {
        Count {
            input: Rc::new(input),
        }
    }

    fn to_field(&self, input: &LogicalPlan) -> Field {
        self.input
            .to_field(input)
            .with_data_type(arrow::datatypes::DataType::UInt64)
    }
}

impl std::fmt::Display for Count {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "COUNT({})", self.input)
    }
}

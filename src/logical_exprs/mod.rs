pub mod aggregate;
pub mod binary;
pub mod column;
pub mod literal;

use crate::logical_plans::LogicalPlan;
use aggregate::AggregateExpr;
use arrow::datatypes::Field;
use binary::Binary;
use column::Column;
use literal::Literal;

#[derive(Clone)]
pub enum LogicalExpr {
    Column(Column),
    Literal(Literal),
    Binary(Binary),
    Aggregate(AggregateExpr),
}

impl LogicalExpr {
    pub fn col(name: impl Into<String>) -> Self {
        LogicalExpr::Column(column::Column::new(name))
    }

    pub fn lit_str(value: impl Into<String>) -> Self {
        LogicalExpr::Literal(literal::Literal::str(value))
    }

    pub fn lit_long(value: i64) -> Self {
        LogicalExpr::Literal(literal::Literal::long(value))
    }

    pub fn to_field(&self, input: &LogicalPlan) -> Field {
        match self {
            LogicalExpr::Column(column) => column.to_field(input),
            LogicalExpr::Literal(literal) => literal.to_field(input),
            LogicalExpr::Binary(binary) => binary.to_field(input),
            LogicalExpr::Aggregate(aggregate) => aggregate.to_field(input),
        }
    }

    pub fn gt(&self, age: LogicalExpr) -> Self {
        LogicalExpr::Binary(Binary::gt(self.clone(), age))
    }

    pub fn gte(&self, age: LogicalExpr) -> Self {
        LogicalExpr::Binary(Binary::gte(self.clone(), age))
    }

    pub fn lt(&self, age: LogicalExpr) -> Self {
        LogicalExpr::Binary(Binary::lt(self.clone(), age))
    }

    pub fn lteq(&self, age: LogicalExpr) -> Self {
        LogicalExpr::Binary(Binary::lteq(self.clone(), age))
    }

    pub fn eq(&self, age: LogicalExpr) -> Self {
        LogicalExpr::Binary(Binary::eq(self.clone(), age))
    }

    pub fn neq(&self, age: LogicalExpr) -> Self {
        LogicalExpr::Binary(Binary::neq(self.clone(), age))
    }

    pub fn and(&self, age: LogicalExpr) -> Self {
        LogicalExpr::Binary(Binary::and(self.clone(), age))
    }

    pub fn or(&self, age: LogicalExpr) -> Self {
        LogicalExpr::Binary(Binary::or(self.clone(), age))
    }
}

impl std::fmt::Display for LogicalExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LogicalExpr::Column(column) => column.fmt(f),
            LogicalExpr::Literal(literal) => literal.fmt(f),
            LogicalExpr::Binary(binary) => binary.fmt(f),
            LogicalExpr::Aggregate(aggregate) => aggregate.fmt(f),
        }
    }
}

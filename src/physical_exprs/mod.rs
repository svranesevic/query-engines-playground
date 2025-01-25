pub mod aggregate;
pub mod binary;
pub mod column;
pub mod literal;

use crate::logical_exprs::binary::Operator;
use aggregate::AggregateExpr;
use arrow::array::{ArrayRef, RecordBatch};
use binary::Binary;
use column::Column;
use literal::Literal;

pub enum PhysicalExpr {
    Column(Column),
    Literal(Literal),
    Binary(Binary),
    Aggregate(AggregateExpr),
}

impl PhysicalExpr {
    pub fn column(i: usize) -> Self {
        PhysicalExpr::Column(Column::new(i))
    }

    pub fn lit_str(value: impl Into<String>) -> Self {
        PhysicalExpr::Literal(Literal::Str(value.into()))
    }

    pub fn lit_long(value: i64) -> Self {
        PhysicalExpr::Literal(Literal::Long(value))
    }

    pub fn gt(left: PhysicalExpr, right: PhysicalExpr) -> Self {
        PhysicalExpr::Binary(Binary::new(left, right, Operator::Gt))
    }

    pub fn gteq(left: PhysicalExpr, right: PhysicalExpr) -> Self {
        PhysicalExpr::Binary(Binary::new(left, right, Operator::GtEq))
    }

    pub fn lt(left: PhysicalExpr, right: PhysicalExpr) -> Self {
        PhysicalExpr::Binary(Binary::new(left, right, Operator::Lt))
    }

    pub fn lteq(left: PhysicalExpr, right: PhysicalExpr) -> Self {
        PhysicalExpr::Binary(Binary::new(left, right, Operator::LtEq))
    }

    pub fn eq(left: PhysicalExpr, right: PhysicalExpr) -> Self {
        PhysicalExpr::Binary(Binary::new(left, right, Operator::Eq))
    }

    pub fn neq(left: PhysicalExpr, right: PhysicalExpr) -> Self {
        PhysicalExpr::Binary(Binary::new(left, right, Operator::NotEq))
    }

    pub fn and(left: PhysicalExpr, right: PhysicalExpr) -> Self {
        PhysicalExpr::Binary(Binary::new(left, right, Operator::And))
    }

    pub fn or(left: PhysicalExpr, right: PhysicalExpr) -> Self {
        PhysicalExpr::Binary(Binary::new(left, right, Operator::Or))
    }

    pub fn evaluate(&self, input: &RecordBatch) -> ArrayRef {
        match self {
            PhysicalExpr::Column(col) => col.evaluate(input),
            PhysicalExpr::Literal(lit) => lit.evaluate(input),
            PhysicalExpr::Binary(binary) => binary.evaluate(input),
            PhysicalExpr::Aggregate(agg) => agg.expression().evaluate(input),
        }
    }
}

impl std::fmt::Display for PhysicalExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PhysicalExpr::Column(col) => col.fmt(f),
            PhysicalExpr::Literal(lit) => lit.fmt(f),
            PhysicalExpr::Binary(binary) => binary.fmt(f),
            PhysicalExpr::Aggregate(agg) => agg.fmt(f),
        }
    }
}

use arrow::datatypes::{DataType, Field};

use crate::logical_plans::LogicalPlan;

use super::LogicalExpr;

#[derive(Clone, Debug)]
pub enum Operator {
    Gt,
    GtEq,
    Lt,
    LtEq,
    Eq,
    NotEq,
    And,
    Or,
}

impl Operator {
    fn result_datatype(&self) -> DataType {
        match self {
            Operator::Gt => DataType::Boolean,
            Operator::GtEq => DataType::Boolean,
            Operator::Lt => DataType::Boolean,
            Operator::LtEq => DataType::Boolean,
            Operator::Eq => DataType::Boolean,
            Operator::NotEq => DataType::Boolean,
            Operator::And => DataType::Boolean,
            Operator::Or => DataType::Boolean,
        }
    }
}

impl std::fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Operator::Gt => f.write_str(">"),
            Operator::GtEq => f.write_str(">="),
            Operator::Lt => f.write_str("<"),
            Operator::LtEq => f.write_str("<="),
            Operator::Eq => f.write_str("=="),
            Operator::NotEq => f.write_str("!="),
            Operator::And => f.write_str("AND"),
            Operator::Or => f.write_str("OR"),
        }
    }
}

#[derive(Clone)]
pub struct Binary {
    pub name: String,
    pub op: Operator,
    pub left: Box<LogicalExpr>,
    pub right: Box<LogicalExpr>,
}

impl Binary {
    pub fn gt(left: LogicalExpr, right: LogicalExpr) -> Self {
        Self {
            name: "gt".to_string(),
            op: Operator::Gt,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn gte(left: LogicalExpr, right: LogicalExpr) -> Self {
        Self {
            name: "gteq".to_string(),
            op: Operator::GtEq,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn lt(left: LogicalExpr, right: LogicalExpr) -> Self {
        Self {
            name: "lt".to_string(),
            op: Operator::Lt,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn lteq(left: LogicalExpr, right: LogicalExpr) -> Self {
        Self {
            name: "lteq".to_string(),
            op: Operator::LtEq,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn eq(left: LogicalExpr, right: LogicalExpr) -> Self {
        Self {
            name: "eq".to_string(),
            op: Operator::Eq,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn neq(left: LogicalExpr, right: LogicalExpr) -> Self {
        Self {
            name: "neq".to_string(),
            op: Operator::NotEq,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn and(left: LogicalExpr, right: LogicalExpr) -> Self {
        Self {
            name: "and".to_string(),
            op: Operator::And,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn or(left: LogicalExpr, right: LogicalExpr) -> Self {
        Self {
            name: "or".to_string(),
            op: Operator::Or,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn to_field(&self, input: &LogicalPlan) -> Field {
        let left_field = self.left.to_field(input);
        let right_field = self.right.to_field(input);

        Field::new(
            &self.name,
            self.op.result_datatype(),
            left_field.is_nullable() || right_field.is_nullable(),
        )
    }
}

impl std::fmt::Display for Binary {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {} {}", self.left, self.op, self.right)
    }
}

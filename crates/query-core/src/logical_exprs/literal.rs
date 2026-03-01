use crate::logical_plans::LogicalPlan;
use arrow::datatypes::{DataType, Field};

#[derive(Clone)]
pub enum Literal {
    Str(String),
    Long(i64),
}

impl Literal {
    pub fn str(value: impl Into<String>) -> Self {
        Literal::Str(value.into())
    }

    pub fn long(value: i64) -> Self {
        Literal::Long(value)
    }

    pub fn to_field(&self, _input: &LogicalPlan) -> Field {
        match self {
            Literal::Str(str) => Field::new(str.clone(), DataType::Utf8, true),
            Literal::Long(i64) => Field::new(i64.to_string(), DataType::Int64, true),
        }
    }
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Literal::Str(str) => write!(f, "'{}'", str),
            Literal::Long(i64) => write!(f, "{}", i64),
        }
    }
}

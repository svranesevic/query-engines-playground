use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use std::sync::Arc;

#[derive(Debug)]
pub enum Literal {
    Str(String),
    Long(i64),
}

impl Literal {
    pub fn evaluate(&self, input: &RecordBatch) -> ArrayRef {
        match self {
            Literal::Str(str) => {
                let value = vec![str.as_str(); input.num_rows()];
                Arc::new(StringArray::from(value))
            }
            Literal::Long(long) => {
                let value = vec![*long; input.num_rows()];
                Arc::new(Int64Array::from(value))
            }
        }
    }
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Literal::Str(str) => write!(f, "'{}'", str),
            Literal::Long(long) => write!(f, "{}", long),
        }
    }
}

use arrow::array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray, UInt64Array};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Literal {
    Str(String),
    Long(i64),
    Double(f64),
    UInt64(u64),
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
            Literal::Double(double) => {
                let value = vec![*double; input.num_rows()];
                Arc::new(Float64Array::from(value))
            }
            Literal::UInt64(uint) => {
                let value = vec![*uint; input.num_rows()];
                Arc::new(UInt64Array::from(value))
            }
        }
    }
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Literal::Str(str) => write!(f, "'{}'", str),
            Literal::Long(long) => write!(f, "{}", long),
            Literal::Double(double) => write!(f, "{}", double),
            Literal::UInt64(uint) => write!(f, "{}", uint),
        }
    }
}

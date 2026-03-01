use arrow::array::{ArrayRef, RecordBatch};

#[derive(Clone)]
pub struct Column {
    pub i: usize,
}

impl Column {
    pub fn new(i: usize) -> Self {
        Self { i }
    }

    pub fn evaluate(&self, input: &RecordBatch) -> ArrayRef {
        input.column(self.i).clone()
    }
}

impl std::fmt::Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "#{}", self.i)
    }
}

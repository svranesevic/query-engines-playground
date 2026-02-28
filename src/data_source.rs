use arrow::{
    array::{Int64Array, RecordBatch, StringViewArray},
    datatypes::{Field, Schema},
};
use std::sync::Arc;

pub trait DataSource {
    fn schema(&self) -> Schema;
    fn scan(&self, projection: Vec<String>) -> Vec<RecordBatch>;
}

pub struct CsvDataSource {
    path: String,
}
impl CsvDataSource {
    pub fn new(path: String) -> Self {
        CsvDataSource { path }
    }
}

impl DataSource for CsvDataSource {
    fn schema(&self) -> Schema {
        // TODO(Sava): Temp
        let fields = vec![
            Field::new("first_name", arrow::datatypes::DataType::Utf8View, false),
            Field::new("last_name", arrow::datatypes::DataType::Utf8View, false),
            Field::new("age", arrow::datatypes::DataType::Int64, false),
        ];
        Schema::new(fields)
    }

    fn scan(&self, projection: Vec<String>) -> Vec<RecordBatch> {
        let schema = self.schema();
        let full_batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(StringViewArray::from(vec![
                    "Alice", "Bob", "Charlie", "Diana", "Eve",
                ])),
                Arc::new(StringViewArray::from(vec![
                    "Smith", "Johnson", "Brown", "Williams", "Davis",
                ])),
                Arc::new(Int64Array::from(vec![25, 42, 19, 42, 30])),
            ],
        )
        .unwrap();

        if projection.is_empty() {
            return vec![full_batch];
        }

        let mut indices = vec![];
        for p in projection {
            if let Ok(index) = schema.index_of(&p) {
                indices.push(index);
            }
        }

        vec![full_batch.project(&indices).unwrap()]
    }
}

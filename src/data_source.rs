use arrow::{
    array::RecordBatch,
    datatypes::{Field, Schema},
};

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

    fn scan(&self, _projection: Vec<String>) -> Vec<RecordBatch> {
        todo!();
    }
}

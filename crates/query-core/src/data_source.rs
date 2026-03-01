use arrow::{
    array::{Int64Array, RecordBatch, StringViewArray},
    csv::ReaderBuilder,
    datatypes::{Field, Schema},
};
use std::{fs::File, path::Path, sync::Arc};

pub trait DataSource {
    fn schema(&self) -> Schema;
    fn scan(&self, projection: Vec<String>) -> Vec<RecordBatch>;
    fn scan_stream(
        &self,
        projection: Vec<String>,
        _batch_size: usize,
    ) -> Box<dyn Iterator<Item = RecordBatch>> {
        Box::new(self.scan(projection).into_iter())
    }
}

pub struct CsvDataSource {
    path: String,
}
impl CsvDataSource {
    pub fn new(path: String) -> Self {
        CsvDataSource { path }
    }

    fn synthetic_scan(&self, projection: Vec<String>) -> Vec<RecordBatch> {
        let schema = self.schema();
        let full_batch = synthetic_batch(schema.clone());

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

    fn project_batch(batch: RecordBatch, projection: &[String]) -> RecordBatch {
        if projection.is_empty() {
            return batch;
        }

        let schema = batch.schema();
        let mut indices = vec![];
        for p in projection {
            if let Ok(index) = schema.index_of(p) {
                indices.push(index);
            }
        }
        batch.project(&indices).unwrap()
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
        if !Path::new(&self.path).exists() {
            return self.synthetic_scan(projection);
        }

        self.scan_stream(projection, 65_536).collect()
    }

    fn scan_stream(
        &self,
        projection: Vec<String>,
        batch_size: usize,
    ) -> Box<dyn Iterator<Item = RecordBatch>> {
        if !Path::new(&self.path).exists() {
            return Box::new(self.synthetic_scan(projection).into_iter());
        }

        let schema = Arc::new(self.schema());
        let projection_for_map = projection.clone();
        let file = File::open(&self.path).unwrap();
        let reader = ReaderBuilder::new(schema)
            .with_header(true)
            .with_batch_size(batch_size)
            .build(file)
            .unwrap();

        Box::new(reader.map(move |batch| {
            let batch = batch.unwrap();
            CsvDataSource::project_batch(batch, &projection_for_map)
        }))
    }
}

fn synthetic_batch(schema: Schema) -> RecordBatch {
    const ROWS: usize = 20_000;
    const FIRST: [&str; 5] = ["Alice", "Bob", "Charlie", "Diana", "Eve"];
    const LAST: [&str; 5] = ["Smith", "Johnson", "Brown", "Williams", "Davis"];
    const AGE: [i64; 5] = [25, 42, 19, 42, 30];

    let first_name = (0..ROWS)
        .map(|i| FIRST[i % FIRST.len()])
        .collect::<Vec<_>>();
    let last_name = (0..ROWS).map(|i| LAST[i % LAST.len()]).collect::<Vec<_>>();
    let age = (0..ROWS).map(|i| AGE[i % AGE.len()]).collect::<Vec<_>>();
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringViewArray::from(first_name)),
            Arc::new(StringViewArray::from(last_name)),
            Arc::new(Int64Array::from(age)),
        ],
    )
    .unwrap()
}

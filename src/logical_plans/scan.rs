use std::rc::Rc;

use super::LogicalPlan;
use crate::data_source::DataSource;
use arrow::datatypes::Schema;

pub struct Scan {
    pub path: String,
    pub data_source: Rc<dyn DataSource>,
    pub projection: Vec<String>,
}

impl Scan {
    pub fn new(path: String, data_source: Rc<dyn DataSource>, projection: Vec<String>) -> Self {
        Self {
            path,
            data_source,
            projection,
        }
    }

    pub fn schema(&self) -> Schema {
        let schema = self.data_source.schema();
        if self.projection.is_empty() {
            return schema;
        }

        let mut indices = vec![];
        for p in &self.projection {
            if let Ok(index) = schema.index_of(p) {
                indices.push(index);
            }
        }
        // Safety: Indices are valid as we got them from the schema above
        schema.project(&indices).unwrap()
    }

    pub fn children(&self) -> Vec<&LogicalPlan> {
        vec![]
    }
}

impl std::fmt::Display for Scan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.projection.is_empty() {
            write!(f, "Scan: {}; projection=None", self.path)
        } else {
            write!(f, "Scan: {}; projection={:?}", self.path, self.projection)
        }
    }
}

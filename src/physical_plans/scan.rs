use std::rc::Rc;

use super::PhysicalPlan;
use crate::data_source::DataSource;
use arrow::{array::RecordBatch, datatypes::Schema};

pub struct ScanExec {
    datasource: Rc<dyn DataSource>,
    projection: Vec<String>,
}

impl ScanExec {
    pub fn new(datasource: Rc<dyn DataSource>, projection: Vec<String>) -> Self {
        Self {
            datasource,
            projection,
        }
    }

    pub fn children(&self) -> Vec<&PhysicalPlan> {
        vec![]
    }

    pub fn schema(&self) -> Schema {
        let schema = self.datasource.schema();
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

    pub fn execute(&mut self) -> Vec<RecordBatch> {
        self.datasource.scan(self.projection.clone())
    }
}

impl std::fmt::Display for ScanExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.projection.is_empty() {
            write!(f, "ScanExec: schema={}, projection=None", self.schema())
        } else {
            write!(
                f,
                "ScanExec: schema={}, projection={:?}",
                self.schema(),
                self.projection
            )
        }
    }
}

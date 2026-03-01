use crate::logical_plans::LogicalPlan;
use arrow::datatypes::Field;

#[derive(Clone)]
pub struct Column {
    pub name: String,
}

impl Column {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn to_field(&self, input: &LogicalPlan) -> Field {
        let schema = input.schema();
        let field = schema
            .field_with_name(&self.name)
            .unwrap_or_else(|_| panic!("no column named {}", self.name));
        field.clone()
    }
}

impl std::fmt::Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "#{}", &self.name)
    }
}

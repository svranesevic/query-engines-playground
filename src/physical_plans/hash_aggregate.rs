use arrow::{
    array::{
        ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
        StringViewArray, UInt64Array,
    },
    datatypes::{DataType, Schema},
};
use std::{collections::HashMap, sync::Arc};

use crate::physical_exprs::{
    aggregate::{Accumulator, AggregateExpr},
    literal::Literal,
    PhysicalExpr,
};

use super::PhysicalPlan;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum GroupValue {
    Int64(i64),
    Utf8(String),
    Bool(bool),
}

type GroupKey = Vec<GroupValue>;

fn make_group_key(group_arrays: &[ArrayRef], row_idx: usize) -> GroupKey {
    group_arrays
        .iter()
        .map(|array| {
            if array.is_null(row_idx) {
                panic!("Nulls in group keys are not supported");
            }
            match array.data_type() {
                DataType::Int64 => {
                    let array = array.as_any().downcast_ref::<Int64Array>().unwrap();
                    GroupValue::Int64(array.value(row_idx))
                }
                DataType::Utf8 => {
                    let array = array.as_any().downcast_ref::<StringArray>().unwrap();
                    GroupValue::Utf8(array.value(row_idx).to_string())
                }
                DataType::Utf8View => {
                    let array = array.as_any().downcast_ref::<StringViewArray>().unwrap();
                    GroupValue::Utf8(array.value(row_idx).to_string())
                }
                DataType::Boolean => {
                    let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                    GroupValue::Bool(array.value(row_idx))
                }
                data_type => panic!("Unsupported group key data type: {:?}", data_type),
            }
        })
        .collect()
}

fn group_columns_from_keys(keys: &[GroupKey], schema: &Schema, group_len: usize) -> Vec<ArrayRef> {
    (0..group_len)
        .map(|col_idx| {
            let field = schema.field(col_idx);
            match field.data_type() {
                DataType::Int64 => {
                    let values: Vec<Option<i64>> = keys
                        .iter()
                        .map(|key| match &key[col_idx] {
                            GroupValue::Int64(v) => Some(*v),
                            other => {
                                panic!("Group key type mismatch at column {}: {:?}", col_idx, other)
                            }
                        })
                        .collect();
                    Arc::new(Int64Array::from(values)) as ArrayRef
                }
                DataType::Utf8 => {
                    let values: Vec<Option<&str>> = keys
                        .iter()
                        .map(|key| match &key[col_idx] {
                            GroupValue::Utf8(v) => Some(v.as_str()),
                            other => {
                                panic!("Group key type mismatch at column {}: {:?}", col_idx, other)
                            }
                        })
                        .collect();
                    Arc::new(StringArray::from(values)) as ArrayRef
                }
                DataType::Utf8View => {
                    let values: Vec<Option<&str>> = keys
                        .iter()
                        .map(|key| match &key[col_idx] {
                            GroupValue::Utf8(v) => Some(v.as_str()),
                            other => {
                                panic!("Group key type mismatch at column {}: {:?}", col_idx, other)
                            }
                        })
                        .collect();
                    Arc::new(StringViewArray::from(values)) as ArrayRef
                }
                DataType::Boolean => {
                    let values: Vec<Option<bool>> = keys
                        .iter()
                        .map(|key| match &key[col_idx] {
                            GroupValue::Bool(v) => Some(*v),
                            other => {
                                panic!("Group key type mismatch at column {}: {:?}", col_idx, other)
                            }
                        })
                        .collect();
                    Arc::new(BooleanArray::from(values)) as ArrayRef
                }
                data_type => panic!("Unsupported group output data type: {:?}", data_type),
            }
        })
        .collect()
}

fn aggregate_columns_from_map(
    key_order: &[GroupKey],
    map: &HashMap<GroupKey, Vec<Box<dyn Accumulator>>>,
    schema: &Schema,
    group_len: usize,
) -> Vec<ArrayRef> {
    (0..(schema.fields().len() - group_len))
        .map(|agg_idx| {
            let field = schema.field(group_len + agg_idx);
            match field.data_type() {
                DataType::Int64 => {
                    let values: Vec<Option<i64>> = key_order
                        .iter()
                        .map(|key| {
                            let accumulators = map.get(key).unwrap();
                            match accumulators[agg_idx].final_value() {
                                Some(Literal::Long(v)) => Some(v),
                                Some(other) => {
                                    panic!(
                                        "Unexpected aggregate literal type for Int64: {:?}",
                                        other
                                    )
                                }
                                None => None,
                            }
                        })
                        .collect();
                    Arc::new(Int64Array::from(values)) as ArrayRef
                }
                DataType::UInt64 => {
                    let values: Vec<Option<u64>> = key_order
                        .iter()
                        .map(|key| {
                            let accumulators = map.get(key).unwrap();
                            match accumulators[agg_idx].final_value() {
                                Some(Literal::UInt64(v)) => Some(v),
                                Some(other) => {
                                    panic!(
                                        "Unexpected aggregate literal type for UInt64: {:?}",
                                        other
                                    )
                                }
                                None => None,
                            }
                        })
                        .collect();
                    Arc::new(UInt64Array::from(values)) as ArrayRef
                }
                DataType::Float64 => {
                    let values: Vec<Option<f64>> = key_order
                        .iter()
                        .map(|key| {
                            let accumulators = map.get(key).unwrap();
                            match accumulators[agg_idx].final_value() {
                                Some(Literal::Double(v)) => Some(v),
                                Some(other) => {
                                    panic!(
                                        "Unexpected aggregate literal type for Float64: {:?}",
                                        other
                                    )
                                }
                                None => None,
                            }
                        })
                        .collect();
                    Arc::new(Float64Array::from(values)) as ArrayRef
                }
                data_type => panic!("Unsupported aggregate output data type: {:?}", data_type),
            }
        })
        .collect()
}

pub struct HashAggregateExec {
    input: Box<PhysicalPlan>,
    group_expr: Vec<PhysicalExpr>,
    aggregate_expr: Vec<AggregateExpr>,
    schema: Schema,
}

impl HashAggregateExec {
    pub fn new(
        input: Box<PhysicalPlan>,
        group_expr: Vec<PhysicalExpr>,
        aggregate_expr: Vec<AggregateExpr>,
        schema: Schema,
    ) -> Self {
        Self {
            input,
            group_expr,
            aggregate_expr,
            schema,
        }
    }

    pub fn schema(&self) -> Schema {
        self.schema.clone()
    }

    pub fn children(&self) -> Vec<&PhysicalPlan> {
        vec![&self.input]
    }

    pub fn execute(&mut self) -> Vec<RecordBatch> {
        let batches = self.input.execute();
        let mut map: HashMap<GroupKey, Vec<Box<dyn Accumulator>>> = HashMap::new();
        let mut key_order = Vec::new();

        for batch in batches {
            let group_arrays: Vec<_> = self.group_expr.iter().map(|e| e.evaluate(&batch)).collect();
            let agg_input_arrays: Vec<_> = self
                .aggregate_expr
                .iter()
                .map(|e| e.expression().evaluate(&batch))
                .collect();

            for row_idx in 0..batch.num_rows() {
                let group_key = make_group_key(&group_arrays, row_idx);
                let accumulators = map.entry(group_key.clone()).or_insert_with(|| {
                    key_order.push(group_key);
                    self.aggregate_expr
                        .iter()
                        .map(AggregateExpr::create_accumulator)
                        .collect()
                });

                for (acc, input_array) in accumulators.iter_mut().zip(agg_input_arrays.iter()) {
                    acc.accumulate(input_array.slice(row_idx, 1));
                }
            }
        }

        if key_order.is_empty() {
            return vec![];
        }

        let group_len = self.group_expr.len();
        let mut columns = group_columns_from_keys(&key_order, &self.schema, group_len);
        columns.extend(aggregate_columns_from_map(
            &key_order,
            &map,
            &self.schema,
            group_len,
        ));

        let batch = RecordBatch::try_new(Arc::new(self.schema.clone()), columns).unwrap();
        vec![batch]
    }
}

impl std::fmt::Display for HashAggregateExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let group_expr = self
            .group_expr
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>();
        let aggregate_expr = self
            .aggregate_expr
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>();
        write!(
            f,
            "HashAggregateExec: groupBy=[{:?}], aggr=[{:?}]",
            group_expr, aggregate_expr
        )
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use arrow::{
        array::{Float64Array, Int64Array, StringArray, UInt64Array},
        datatypes::{DataType, Field},
    };

    use crate::{data_source::DataSource, physical_exprs::aggregate::AggregateExpr};

    use super::*;

    #[derive(Clone)]
    struct TestDataSource {
        schema: Schema,
        batches: Vec<RecordBatch>,
    }

    impl TestDataSource {
        fn new(schema: Schema, batches: Vec<RecordBatch>) -> Self {
            Self { schema, batches }
        }
    }

    impl DataSource for TestDataSource {
        fn schema(&self) -> Schema {
            self.schema.clone()
        }

        fn scan(&self, _projection: Vec<String>) -> Vec<RecordBatch> {
            self.batches.clone()
        }
    }

    fn execute_hash_agg(
        input_schema: Schema,
        input_batches: Vec<RecordBatch>,
        group_expr: Vec<PhysicalExpr>,
        aggregate_expr: Vec<AggregateExpr>,
        output_schema: Schema,
    ) -> Vec<RecordBatch> {
        let data_source = Rc::new(TestDataSource::new(input_schema, input_batches));
        let input = PhysicalPlan::scan(data_source, vec![]);
        let mut agg =
            HashAggregateExec::new(Box::new(input), group_expr, aggregate_expr, output_schema);
        agg.execute()
    }

    #[test]
    fn group_by_single_int64_max() {
        let input_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]);
        let input = RecordBatch::try_new(
            Arc::new(input_schema.clone()),
            vec![
                Arc::new(Int64Array::from(vec![1, 1, 2, 2])),
                Arc::new(Int64Array::from(vec![10, 30, 20, 15])),
            ],
        )
        .unwrap();

        let output_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("max_value", DataType::Int64, true),
        ]);
        let result = execute_hash_agg(
            input_schema,
            vec![input],
            vec![PhysicalExpr::column(0)],
            vec![AggregateExpr::max(Box::new(PhysicalExpr::column(1)))],
            output_schema,
        );

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        let groups = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let max_vals = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(groups.values(), &[1, 2]);
        assert_eq!(max_vals.values(), &[30, 20]);
    }

    #[test]
    fn group_by_utf8_max_int64() {
        let input_schema = Schema::new(vec![
            Field::new("group_key", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
        ]);
        let input = RecordBatch::try_new(
            Arc::new(input_schema.clone()),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "a"])),
                Arc::new(Int64Array::from(vec![5, 3, 7])),
            ],
        )
        .unwrap();

        let output_schema = Schema::new(vec![
            Field::new("group_key", DataType::Utf8, false),
            Field::new("max_value", DataType::Int64, true),
        ]);
        let result = execute_hash_agg(
            input_schema,
            vec![input],
            vec![PhysicalExpr::column(0)],
            vec![AggregateExpr::max(Box::new(PhysicalExpr::column(1)))],
            output_schema,
        );

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        let groups = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let max_vals = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(groups.value(0), "a");
        assert_eq!(groups.value(1), "b");
        assert_eq!(max_vals.values(), &[7, 3]);
    }

    #[test]
    fn empty_input_returns_no_batches() {
        let input_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]);
        let input = RecordBatch::try_new(
            Arc::new(input_schema.clone()),
            vec![
                Arc::new(Int64Array::from(Vec::<i64>::new())),
                Arc::new(Int64Array::from(Vec::<i64>::new())),
            ],
        )
        .unwrap();

        let output_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("max_value", DataType::Int64, true),
        ]);
        let result = execute_hash_agg(
            input_schema,
            vec![input],
            vec![PhysicalExpr::column(0)],
            vec![AggregateExpr::max(Box::new(PhysicalExpr::column(1)))],
            output_schema,
        );

        assert!(result.is_empty());
    }

    #[test]
    fn deterministic_group_order() {
        let input_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]);
        let input = RecordBatch::try_new(
            Arc::new(input_schema.clone()),
            vec![
                Arc::new(Int64Array::from(vec![2, 1, 2, 3, 1])),
                Arc::new(Int64Array::from(vec![5, 8, 10, 4, 7])),
            ],
        )
        .unwrap();

        let output_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("max_value", DataType::Int64, true),
        ]);
        let result = execute_hash_agg(
            input_schema,
            vec![input],
            vec![PhysicalExpr::column(0)],
            vec![AggregateExpr::max(Box::new(PhysicalExpr::column(1)))],
            output_schema,
        );

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        let groups = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let max_vals = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(groups.values(), &[2, 1, 3]);
        assert_eq!(max_vals.values(), &[10, 8, 4]);
    }

    #[test]
    fn group_by_single_int64_sum() {
        let input_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]);
        let input = RecordBatch::try_new(
            Arc::new(input_schema.clone()),
            vec![
                Arc::new(Int64Array::from(vec![1, 1, 2, 2])),
                Arc::new(Int64Array::from(vec![10, 30, 20, 15])),
            ],
        )
        .unwrap();

        let output_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("sum_value", DataType::Int64, true),
        ]);
        let result = execute_hash_agg(
            input_schema,
            vec![input],
            vec![PhysicalExpr::column(0)],
            vec![AggregateExpr::sum(Box::new(PhysicalExpr::column(1)))],
            output_schema,
        );

        let batch = &result[0];
        let groups = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let sums = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(groups.values(), &[1, 2]);
        assert_eq!(sums.values(), &[40, 35]);
    }

    #[test]
    fn group_by_single_int64_min() {
        let input_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]);
        let input = RecordBatch::try_new(
            Arc::new(input_schema.clone()),
            vec![
                Arc::new(Int64Array::from(vec![1, 1, 2, 2])),
                Arc::new(Int64Array::from(vec![10, 30, 20, 15])),
            ],
        )
        .unwrap();

        let output_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("min_value", DataType::Int64, true),
        ]);
        let result = execute_hash_agg(
            input_schema,
            vec![input],
            vec![PhysicalExpr::column(0)],
            vec![AggregateExpr::min(Box::new(PhysicalExpr::column(1)))],
            output_schema,
        );

        let batch = &result[0];
        let groups = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let mins = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(groups.values(), &[1, 2]);
        assert_eq!(mins.values(), &[10, 15]);
    }

    #[test]
    fn group_by_single_int64_avg() {
        let input_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]);
        let input = RecordBatch::try_new(
            Arc::new(input_schema.clone()),
            vec![
                Arc::new(Int64Array::from(vec![1, 1, 2, 2])),
                Arc::new(Int64Array::from(vec![1, 2, 10, 20])),
            ],
        )
        .unwrap();

        let output_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("avg_value", DataType::Float64, true),
        ]);
        let result = execute_hash_agg(
            input_schema,
            vec![input],
            vec![PhysicalExpr::column(0)],
            vec![AggregateExpr::avg(Box::new(PhysicalExpr::column(1)))],
            output_schema,
        );

        let batch = &result[0];
        let groups = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let avgs = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(groups.values(), &[1, 2]);
        assert_eq!(avgs.value(0), 1.5);
        assert_eq!(avgs.value(1), 15.0);
    }

    #[test]
    fn group_by_single_int64_count_non_null() {
        let input_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("value", DataType::Int64, true),
        ]);
        let input = RecordBatch::try_new(
            Arc::new(input_schema.clone()),
            vec![
                Arc::new(Int64Array::from(vec![1, 1, 2, 2])),
                Arc::new(Int64Array::from(vec![Some(10), None, Some(20), None])),
            ],
        )
        .unwrap();

        let output_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("count_value", DataType::UInt64, true),
        ]);
        let result = execute_hash_agg(
            input_schema,
            vec![input],
            vec![PhysicalExpr::column(0)],
            vec![AggregateExpr::count(Box::new(PhysicalExpr::column(1)))],
            output_schema,
        );

        let batch = &result[0];
        let groups = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let counts = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(groups.values(), &[1, 2]);
        assert_eq!(counts.values(), &[1, 1]);
    }

    #[test]
    fn mixed_aggregates_in_one_query() {
        let input_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("value", DataType::Int64, true),
        ]);
        let input = RecordBatch::try_new(
            Arc::new(input_schema.clone()),
            vec![
                Arc::new(Int64Array::from(vec![1, 1, 2, 2])),
                Arc::new(Int64Array::from(vec![Some(10), Some(30), Some(20), None])),
            ],
        )
        .unwrap();

        let output_schema = Schema::new(vec![
            Field::new("group_key", DataType::Int64, false),
            Field::new("min_value", DataType::Int64, true),
            Field::new("max_value", DataType::Int64, true),
            Field::new("sum_value", DataType::Int64, true),
            Field::new("avg_value", DataType::Float64, true),
            Field::new("count_value", DataType::UInt64, true),
        ]);
        let result = execute_hash_agg(
            input_schema,
            vec![input],
            vec![PhysicalExpr::column(0)],
            vec![
                AggregateExpr::min(Box::new(PhysicalExpr::column(1))),
                AggregateExpr::max(Box::new(PhysicalExpr::column(1))),
                AggregateExpr::sum(Box::new(PhysicalExpr::column(1))),
                AggregateExpr::avg(Box::new(PhysicalExpr::column(1))),
                AggregateExpr::count(Box::new(PhysicalExpr::column(1))),
            ],
            output_schema,
        );

        let batch = &result[0];
        let groups = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let mins = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let maxs = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let sums = batch
            .column(3)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let avgs = batch
            .column(4)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let counts = batch
            .column(5)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        assert_eq!(groups.values(), &[1, 2]);
        assert_eq!(mins.values(), &[10, 20]);
        assert_eq!(maxs.values(), &[30, 20]);
        assert_eq!(sums.values(), &[40, 20]);
        assert_eq!(avgs.value(0), 20.0);
        assert_eq!(avgs.value(1), 20.0);
        assert_eq!(counts.values(), &[2, 1]);
    }

    #[test]
    #[should_panic(expected = "Unsupported group key data type")]
    fn unsupported_group_type_panics() {
        let input_schema = Schema::new(vec![
            Field::new("group_key", DataType::Float64, false),
            Field::new("value", DataType::Int64, false),
        ]);
        let input = RecordBatch::try_new(
            Arc::new(input_schema.clone()),
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Int64Array::from(vec![10, 20])),
            ],
        )
        .unwrap();

        let output_schema = Schema::new(vec![
            Field::new("group_key", DataType::Float64, false),
            Field::new("max_value", DataType::Int64, true),
        ]);
        let _ = execute_hash_agg(
            input_schema,
            vec![input],
            vec![PhysicalExpr::column(0)],
            vec![AggregateExpr::max(Box::new(PhysicalExpr::column(1)))],
            output_schema,
        );
    }
}

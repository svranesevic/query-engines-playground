use std::sync::Arc;

use arrow::{
    array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray, StringViewArray},
    compute::kernels,
    datatypes::DataType,
};

use query_core::logical_exprs::binary::Operator;

use super::PhysicalExpr;

#[derive(Clone)]
pub struct Binary {
    left: Box<PhysicalExpr>,
    right: Box<PhysicalExpr>,
    op: Operator,
}

impl Binary {
    pub fn new(left: PhysicalExpr, right: PhysicalExpr, op: Operator) -> Self {
        Self {
            left: Box::new(left),
            right: Box::new(right),
            op,
        }
    }

    pub fn evaluate(&self, input: &RecordBatch) -> ArrayRef {
        let left = self.left.evaluate(input);
        let right = self.right.evaluate(input);

        if left.data_type() != right.data_type() {
            panic!(
                "Type mismatch in binary expression: left={:?}, right={:?}",
                left.data_type(),
                right.data_type()
            );
        }

        let compare = |op: Operator| -> arrow::array::BooleanArray {
            match left.data_type() {
                DataType::Int64 => {
                    let left = left.as_any().downcast_ref::<Int64Array>().unwrap();
                    let right = right.as_any().downcast_ref::<Int64Array>().unwrap();
                    match op {
                        Operator::Gt => kernels::cmp::gt(left, right).unwrap(),
                        Operator::GtEq => kernels::cmp::gt_eq(left, right).unwrap(),
                        Operator::Lt => kernels::cmp::lt(left, right).unwrap(),
                        Operator::LtEq => kernels::cmp::lt_eq(left, right).unwrap(),
                        Operator::Eq => kernels::cmp::eq(left, right).unwrap(),
                        Operator::NotEq => kernels::cmp::neq(left, right).unwrap(),
                        _ => unreachable!(),
                    }
                }
                DataType::Utf8 => {
                    let left = left.as_any().downcast_ref::<StringArray>().unwrap();
                    let right = right.as_any().downcast_ref::<StringArray>().unwrap();
                    match op {
                        Operator::Gt => kernels::cmp::gt(left, right).unwrap(),
                        Operator::GtEq => kernels::cmp::gt_eq(left, right).unwrap(),
                        Operator::Lt => kernels::cmp::lt(left, right).unwrap(),
                        Operator::LtEq => kernels::cmp::lt_eq(left, right).unwrap(),
                        Operator::Eq => kernels::cmp::eq(left, right).unwrap(),
                        Operator::NotEq => kernels::cmp::neq(left, right).unwrap(),
                        _ => unreachable!(),
                    }
                }
                DataType::Utf8View => {
                    let left = left.as_any().downcast_ref::<StringViewArray>().unwrap();
                    let right = right.as_any().downcast_ref::<StringViewArray>().unwrap();
                    match op {
                        Operator::Gt => kernels::cmp::gt(left, right).unwrap(),
                        Operator::GtEq => kernels::cmp::gt_eq(left, right).unwrap(),
                        Operator::Lt => kernels::cmp::lt(left, right).unwrap(),
                        Operator::LtEq => kernels::cmp::lt_eq(left, right).unwrap(),
                        Operator::Eq => kernels::cmp::eq(left, right).unwrap(),
                        Operator::NotEq => kernels::cmp::neq(left, right).unwrap(),
                        _ => unreachable!(),
                    }
                }
                DataType::Boolean => {
                    let left = left.as_any().downcast_ref::<BooleanArray>().unwrap();
                    let right = right.as_any().downcast_ref::<BooleanArray>().unwrap();
                    match op {
                        Operator::Eq => kernels::cmp::eq(left, right).unwrap(),
                        Operator::NotEq => kernels::cmp::neq(left, right).unwrap(),
                        _ => panic!("Unsupported comparison operator {:?} for Boolean", op),
                    }
                }
                data_type => panic!("Unsupported comparison data type: {:?}", data_type),
            }
        };

        let array = match self.op {
            Operator::Gt
            | Operator::GtEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Eq
            | Operator::NotEq => compare(self.op.clone()),
            Operator::And => {
                let left = left.as_any().downcast_ref::<BooleanArray>().unwrap();
                let right = right.as_any().downcast_ref::<BooleanArray>().unwrap();
                kernels::boolean::and(left, right).unwrap()
            }
            Operator::Or => {
                let left = left.as_any().downcast_ref::<BooleanArray>().unwrap();
                let right = right.as_any().downcast_ref::<BooleanArray>().unwrap();
                kernels::boolean::or(left, right).unwrap()
            }
        };
        Arc::new(array)
    }

    pub fn left(&self) -> &PhysicalExpr {
        self.left.as_ref()
    }

    pub fn right(&self) -> &PhysicalExpr {
        self.right.as_ref()
    }

    pub fn op(&self) -> &Operator {
        &self.op
    }
}

impl std::fmt::Display for Binary {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {} {}", self.left, self.op, self.right)
    }
}

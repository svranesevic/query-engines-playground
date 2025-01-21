use std::sync::Arc;

use arrow::{
    array::{ArrayRef, BooleanArray, RecordBatch},
    compute::kernels,
};

use crate::logical_exprs::binary::Operator;

use super::PhysicalExpr;

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
        let array = match self.op {
            Operator::Gt => {
                let left = left.as_any().downcast_ref::<BooleanArray>().unwrap();
                let right = right.as_any().downcast_ref::<BooleanArray>().unwrap();
                kernels::cmp::gt(left, right)
            }
            Operator::GtEq => {
                let left = left.as_any().downcast_ref::<BooleanArray>().unwrap();
                let right = right.as_any().downcast_ref::<BooleanArray>().unwrap();
                kernels::cmp::gt_eq(left, right)
            }
            Operator::Lt => {
                let left = left.as_any().downcast_ref::<BooleanArray>().unwrap();
                let right = right.as_any().downcast_ref::<BooleanArray>().unwrap();
                kernels::cmp::lt(left, right)
            }
            Operator::LtEq => {
                let left = left.as_any().downcast_ref::<BooleanArray>().unwrap();
                let right = right.as_any().downcast_ref::<BooleanArray>().unwrap();
                kernels::cmp::lt_eq(left, right)
            }
            Operator::Eq => {
                let left = left.as_any().downcast_ref::<BooleanArray>().unwrap();
                let right = right.as_any().downcast_ref::<BooleanArray>().unwrap();
                kernels::cmp::eq(left, right)
            }
            Operator::NotEq => {
                let left = left.as_any().downcast_ref::<BooleanArray>().unwrap();
                let right = right.as_any().downcast_ref::<BooleanArray>().unwrap();
                kernels::cmp::neq(left, right)
            }
            Operator::And => {
                let left = left.as_any().downcast_ref::<BooleanArray>().unwrap();
                let right = right.as_any().downcast_ref::<BooleanArray>().unwrap();
                kernels::boolean::and(left, right)
            }
            Operator::Or => {
                let left = left.as_any().downcast_ref::<BooleanArray>().unwrap();
                let right = right.as_any().downcast_ref::<BooleanArray>().unwrap();
                kernels::boolean::or(left, right)
            }
        };
        Arc::new(array.unwrap())
    }
}

impl std::fmt::Display for Binary {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {} {}", self.left, self.op, self.right)
    }
}

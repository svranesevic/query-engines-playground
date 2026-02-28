use arrow::{
    array::{ArrayRef, ArrowNativeTypeOp, ArrowNumericType, AsArray},
    compute::kernels::aggregate,
    datatypes::{ArrowNativeType, DataType, Int64Type},
};

use super::{literal::Literal, PhysicalExpr};

pub enum AggregateExpr {
    Max(Max),
    Sum(Sum),
    Min(Min),
    Average(Average),
    Count(Count),
}

impl AggregateExpr {
    pub fn max(expr: Box<PhysicalExpr>) -> Self {
        AggregateExpr::Max(Max::new(expr))
    }

    pub fn sum(expr: Box<PhysicalExpr>) -> Self {
        AggregateExpr::Sum(Sum::new(expr))
    }

    pub fn min(expr: Box<PhysicalExpr>) -> Self {
        AggregateExpr::Min(Min::new(expr))
    }

    pub fn avg(expr: Box<PhysicalExpr>) -> Self {
        AggregateExpr::Average(Average::new(expr))
    }

    pub fn count(expr: Box<PhysicalExpr>) -> Self {
        AggregateExpr::Count(Count::new(expr))
    }

    pub fn expression(&self) -> &PhysicalExpr {
        match self {
            AggregateExpr::Max(max) => max.expression(),
            AggregateExpr::Sum(sum) => sum.expression(),
            AggregateExpr::Min(min) => min.expression(),
            AggregateExpr::Average(average) => average.expression(),
            AggregateExpr::Count(count) => count.expression(),
        }
    }

    pub fn create_accumulator(&self) -> Box<dyn Accumulator> {
        match self {
            AggregateExpr::Max(max) => max.create_accumulator(),
            AggregateExpr::Sum(sum) => sum.create_accumulator(),
            AggregateExpr::Min(min) => min.create_accumulator(),
            AggregateExpr::Average(average) => average.create_accumulator(),
            AggregateExpr::Count(count) => count.create_accumulator(),
        }
    }
}

impl std::fmt::Display for AggregateExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregateExpr::Max(max) => write!(f, "MAX({})", max.expression()),
            AggregateExpr::Sum(sum) => write!(f, "SUM({})", sum.expression()),
            AggregateExpr::Min(min) => write!(f, "MIN({})", min.expression()),
            AggregateExpr::Average(average) => write!(f, "AVG({})", average.expression()),
            AggregateExpr::Count(count) => write!(f, "COUNT({})", count.expression()),
        }
    }
}

pub trait Accumulator {
    fn accumulate(&mut self, values: ArrayRef);
    fn final_value(&self) -> Option<Literal>;
}

pub struct Max {
    expr: Box<PhysicalExpr>,
}

impl Max {
    fn new(expr: Box<PhysicalExpr>) -> Self {
        Self { expr }
    }

    fn expression(&self) -> &PhysicalExpr {
        self.expr.as_ref()
    }

    fn create_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(MaxAccumulator::<Int64Type>::new())
    }
}

struct MaxAccumulator<T: ArrowNumericType> {
    max: Option<T::Native>,
}

impl<T: ArrowNumericType> MaxAccumulator<T> {
    fn new() -> Self {
        Self { max: None }
    }
}

pub struct Sum {
    expr: Box<PhysicalExpr>,
}

impl Sum {
    fn new(expr: Box<PhysicalExpr>) -> Self {
        Self { expr }
    }

    fn expression(&self) -> &PhysicalExpr {
        self.expr.as_ref()
    }

    fn create_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(SumAccumulator::<Int64Type>::new())
    }
}

struct SumAccumulator<T: ArrowNumericType> {
    sum: Option<T::Native>,
}

impl<T: ArrowNumericType> SumAccumulator<T> {
    fn new() -> Self {
        Self { sum: None }
    }
}

impl<T: ArrowNumericType> Accumulator for SumAccumulator<T> {
    fn accumulate(&mut self, values: ArrayRef) {
        let values = values.as_primitive::<T>();
        let Some(sum) = aggregate::sum(values) else {
            return;
        };

        self.sum = match self.sum {
            Some(current) => Some(current.add_wrapping(sum)),
            None => Some(sum),
        };
    }

    fn final_value(&self) -> Option<Literal> {
        match T::DATA_TYPE {
            DataType::Int64 => self.sum.and_then(|v| v.to_i64()).map(Literal::Long),
            _ => panic!("Unsupported data type for sum accumulator"),
        }
    }
}

pub struct Min {
    expr: Box<PhysicalExpr>,
}

impl Min {
    fn new(expr: Box<PhysicalExpr>) -> Self {
        Self { expr }
    }

    fn expression(&self) -> &PhysicalExpr {
        self.expr.as_ref()
    }

    fn create_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(MinAccumulator::<Int64Type>::new())
    }
}

struct MinAccumulator<T: ArrowNumericType> {
    min: Option<T::Native>,
}

impl<T: ArrowNumericType> MinAccumulator<T> {
    fn new() -> Self {
        Self { min: None }
    }
}

impl<T: ArrowNumericType> Accumulator for MinAccumulator<T> {
    fn accumulate(&mut self, values: ArrayRef) {
        let values = values.as_primitive::<T>();
        let Some(min) = aggregate::min(values) else {
            return;
        };

        match self.min {
            Some(current) => {
                if min.is_lt(current) {
                    self.min = Some(min);
                }
            }
            None => self.min = Some(min),
        }
    }

    fn final_value(&self) -> Option<Literal> {
        match T::DATA_TYPE {
            DataType::Int64 => self.min.and_then(|v| v.to_i64()).map(Literal::Long),
            _ => panic!("Unsupported data type for min accumulator"),
        }
    }
}

pub struct Average {
    expr: Box<PhysicalExpr>,
}

impl Average {
    fn new(expr: Box<PhysicalExpr>) -> Self {
        Self { expr }
    }

    fn expression(&self) -> &PhysicalExpr {
        self.expr.as_ref()
    }

    fn create_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(AvgAccumulator::new())
    }
}

struct AvgAccumulator {
    sum: i128,
    count: u64,
}

impl AvgAccumulator {
    fn new() -> Self {
        Self { sum: 0, count: 0 }
    }
}

impl Accumulator for AvgAccumulator {
    fn accumulate(&mut self, values: ArrayRef) {
        if values.data_type() != &DataType::Int64 {
            panic!(
                "Unsupported data type for avg accumulator: {:?}",
                values.data_type()
            );
        }
        let values = values.as_primitive::<Int64Type>();
        for value in values.iter().flatten() {
            self.sum += value as i128;
            self.count += 1;
        }
    }

    fn final_value(&self) -> Option<Literal> {
        if self.count == 0 {
            return None;
        }
        Some(Literal::Double(self.sum as f64 / self.count as f64))
    }
}

pub struct Count {
    expr: Box<PhysicalExpr>,
}

impl Count {
    fn new(expr: Box<PhysicalExpr>) -> Self {
        Self { expr }
    }

    fn expression(&self) -> &PhysicalExpr {
        self.expr.as_ref()
    }

    fn create_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(CountAccumulator::new())
    }
}

struct CountAccumulator {
    count: u64,
}

impl CountAccumulator {
    fn new() -> Self {
        Self { count: 0 }
    }
}

impl Accumulator for CountAccumulator {
    fn accumulate(&mut self, values: ArrayRef) {
        self.count += (values.len() - values.null_count()) as u64;
    }

    fn final_value(&self) -> Option<Literal> {
        Some(Literal::UInt64(self.count))
    }
}

impl<T: ArrowNumericType> Accumulator for MaxAccumulator<T> {
    fn accumulate(&mut self, values: ArrayRef) {
        let values = values.as_primitive::<T>();
        let Some(max) = aggregate::max(values) else {
            return;
        };

        match self.max {
            Some(current) => {
                if max.is_gt(current) {
                    self.max = Some(max);
                }
            }
            None => self.max = Some(max),
        }
    }

    fn final_value(&self) -> Option<Literal> {
        match T::DATA_TYPE {
            DataType::Int64 => self.max.and_then(|v| v.to_i64()).map(Literal::Long),
            _ => panic!("Unsupported data type for max accumulator"),
        }
    }
}

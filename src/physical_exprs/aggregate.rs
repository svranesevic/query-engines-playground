use arrow::array::{ArrayRef, AsArray};

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
        Box::new(MaxAccumulator::default())
    }
}

// TODO(Sava): Use arrow_schema::DataType, to figure out accumulator type
#[derive(Default)]
struct MaxAccumulator {
    value: Option<Literal>,
}

impl Accumulator for MaxAccumulator {
    fn accumulate(&mut self, values: ArrayRef) {
        let values = values
            .as_ref()
            .as_primitive::<arrow::datatypes::Int64Type>();
        let max = arrow::compute::kernels::aggregate::max(values);

        if let Some(max) = max {
            let new_max = match self.value {
                None => Literal::Long(max),
                Some(Literal::Long(old_max)) => Literal::Long(old_max.max(max)),
                // TODO(Sava): Use arrow_schema::DataType, to figure out accumulator type
                _ => panic!(),
            };
            self.value = Some(new_max);
        }
    }

    fn final_value(&self) -> Option<Literal> {
        self.value.clone()
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
        Box::new(SumAccumulator::new())
    }
}

struct SumAccumulator {
    value: Literal,
}

impl SumAccumulator {
    fn new() -> Self {
        Self {
            value: Literal::Long(0),
        }
    }
}

impl Accumulator for SumAccumulator {
    fn accumulate(&mut self, values: ArrayRef) {
        let values = values
            .as_ref()
            .as_primitive::<arrow::datatypes::Int64Type>();
        let sum = arrow::compute::kernels::aggregate::sum(values);

        if let Some(sum) = sum {
            self.value = match self.value {
                Literal::Long(old_sum) => Literal::Long(old_sum + sum),
                _ => panic!(),
            };
        }
    }

    fn final_value(&self) -> Option<Literal> {
        Some(self.value.clone())
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
        Box::new(MinAccumulator::default())
    }
}

#[derive(Default)]
struct MinAccumulator {
    value: Option<Literal>,
}

impl Accumulator for MinAccumulator {
    fn accumulate(&mut self, values: ArrayRef) {
        let values = values
            .as_ref()
            .as_primitive::<arrow::datatypes::Int64Type>();
        let min = arrow::compute::kernels::aggregate::min(values);

        if let Some(min) = min {
            let new_min = match self.value {
                None => Literal::Long(min),
                Some(Literal::Long(old_min)) => Literal::Long(old_min.min(min)),
                _ => panic!(),
            };
            self.value = Some(new_min);
        }
    }

    fn final_value(&self) -> Option<Literal> {
        self.value.clone()
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
        Box::new(AverageAccumulator::new())
    }
}

struct AverageAccumulator {
    sum: Literal,
    count: u64,
}

impl AverageAccumulator {
    fn new() -> Self {
        Self {
            sum: Literal::Long(0),
            count: 0,
        }
    }
}

impl Accumulator for AverageAccumulator {
    fn accumulate(&mut self, values: ArrayRef) {
        let values = values
            .as_ref()
            .as_primitive::<arrow::datatypes::Int64Type>();
        let sum = arrow::compute::kernels::aggregate::sum(values);
        let count = values.len() as u64;

        if let Some(sum) = sum {
            self.sum = match self.sum {
                Literal::Long(old_sum) => Literal::Long(old_sum + sum),
                _ => panic!(),
            };
            self.count += count;
        }
    }

    fn final_value(&self) -> Option<Literal> {
        if self.count == 0 {
            return None;
        }

        let sum = match self.sum {
            Literal::Long(sum) => sum,
            _ => panic!(),
        };
        Some(Literal::Long(sum / self.count as i64))
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
        Box::new(CountAccumulator::default())
    }
}

#[derive(Default)]
pub struct CountAccumulator {
    count: u64,
}

impl Accumulator for CountAccumulator {
    fn accumulate(&mut self, values: ArrayRef) {
        self.count += values.len() as u64;
    }

    fn final_value(&self) -> Option<Literal> {
        Some(Literal::Long(self.count as i64))
    }
}

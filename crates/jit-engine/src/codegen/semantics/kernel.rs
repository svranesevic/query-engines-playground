use crate::lowering::FusedProgram;

use super::looping::{FilterStrategy, RowLoopSpec};

pub(crate) struct SemanticKernel<'a> {
    pub(crate) name: &'a str,
    pub(crate) plan: &'a FusedProgram,
    pub(crate) row_loop: RowLoopSpec,
    pub(crate) filter_strategy: FilterStrategy,
}

pub(crate) struct SemanticKernelBuilder<'a> {
    name: &'a str,
    plan: &'a FusedProgram,
    row_loop: RowLoopSpec,
    filter_strategy: FilterStrategy,
}

impl<'a> SemanticKernelBuilder<'a> {
    pub(crate) fn new(name: &'a str, plan: &'a FusedProgram) -> Self {
        Self {
            name,
            plan,
            row_loop: RowLoopSpec { step: 2 },
            filter_strategy: FilterStrategy::Scalar,
        }
    }

    pub(crate) fn row_loop(mut self, row_loop: RowLoopSpec) -> Self {
        self.row_loop = row_loop;
        self
    }

    pub(crate) fn with_filter_strategy(mut self, strategy: FilterStrategy) -> Self {
        self.filter_strategy = strategy;
        self
    }

    pub(crate) fn finalize(self) -> SemanticKernel<'a> {
        SemanticKernel {
            name: self.name,
            plan: self.plan,
            row_loop: self.row_loop,
            filter_strategy: self.filter_strategy,
        }
    }
}

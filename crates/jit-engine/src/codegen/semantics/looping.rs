#[derive(Clone, Copy, Debug)]
pub(crate) struct RowLoopSpec {
    pub(crate) step: i64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SimdInt64FilterSpec {
    pub(crate) col_idx: i64,
    pub(crate) lit: i64,
    pub(crate) cmp: super::expr::SemanticCmp,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum FilterStrategy {
    Scalar,
    SimdInt64Literal(SimdInt64FilterSpec),
}

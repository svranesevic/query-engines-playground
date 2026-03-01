#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SemType {
    I8,
    I32,
    I64,
    F64,
    Ptr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntCmp {
    Eq,
    NotEq,
    SignedGt,
    SignedGtEq,
    SignedLt,
    SignedLtEq,
    UnsignedLtEq,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FloatCmp {
    Eq,
    NotEq,
    Gt,
    GtEq,
    Lt,
    LtEq,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SemValue(pub(crate) cranelift_codegen::ir::Value);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SemBlock(pub(crate) cranelift_codegen::ir::Block);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SemVar(pub(crate) usize);

pub(crate) trait Backend {
    fn pointer_bytes(&self) -> i64;

    fn create_block(&mut self) -> SemBlock;
    fn append_block_params_for_function_params(&mut self, block: SemBlock);
    fn append_block_param(&mut self, block: SemBlock, ty: SemType);
    fn switch_to_block(&mut self, block: SemBlock);
    fn seal_block(&mut self, block: SemBlock);
    fn block_param(&mut self, block: SemBlock, index: usize) -> SemValue;

    fn jump(&mut self, block: SemBlock, args: &[SemValue]);
    fn brif(
        &mut self,
        cond: SemValue,
        then_block: SemBlock,
        then_args: &[SemValue],
        else_block: SemBlock,
        else_args: &[SemValue],
    );

    fn declare_var(&mut self, var: SemVar, ty: SemType);
    fn def_var(&mut self, var: SemVar, value: SemValue);
    fn use_var(&mut self, var: SemVar) -> SemValue;

    fn iconst_i64(&mut self, v: i64) -> SemValue;
    fn iconst_i8(&mut self, v: i64) -> SemValue;
    fn f64const(&mut self, v: f64) -> SemValue;

    fn iadd(&mut self, a: SemValue, b: SemValue) -> SemValue;
    fn iadd_imm(&mut self, a: SemValue, imm: i64) -> SemValue;
    fn isub(&mut self, a: SemValue, b: SemValue) -> SemValue;
    fn imul_imm(&mut self, a: SemValue, imm: i64) -> SemValue;

    fn icmp(&mut self, cc: IntCmp, a: SemValue, b: SemValue) -> SemValue;
    fn icmp_imm(&mut self, cc: IntCmp, a: SemValue, imm: i64) -> SemValue;
    fn fcmp(&mut self, cc: FloatCmp, a: SemValue, b: SemValue) -> SemValue;

    fn band(&mut self, a: SemValue, b: SemValue) -> SemValue;
    fn bor(&mut self, a: SemValue, b: SemValue) -> SemValue;
    fn bxor_imm(&mut self, a: SemValue, imm: i64) -> SemValue;
    fn select(&mut self, cond_b1: SemValue, yes: SemValue, no: SemValue) -> SemValue;

    fn uextend(&mut self, to_ty: SemType, v: SemValue) -> SemValue;

    fn load(&mut self, ty: SemType, base: SemValue, offset: i32) -> SemValue;
    fn store(&mut self, value: SemValue, base: SemValue, offset: i32);

    fn splat_i64x2(&mut self, scalar_i64: SemValue) -> SemValue;
    fn load_i64x2(&mut self, base: SemValue, offset: i32) -> SemValue;
    fn icmp_i64x2(&mut self, cc: IntCmp, lhs: SemValue, rhs: SemValue) -> SemValue;
    fn extract_lane(&mut self, vec: SemValue, lane: u8) -> SemValue;

    fn call_str_cmp(
        &mut self,
        left_ptr: SemValue,
        left_len: SemValue,
        right_ptr: SemValue,
        right_len: SemValue,
        op_code: SemValue,
    ) -> SemValue;

    fn call_memcmp_eq(
        &mut self,
        left_ptr: SemValue,
        right_ptr: SemValue,
        len: SemValue,
    ) -> SemValue;

    fn return_void(&mut self);
}

pub(crate) mod cranelift;

use cranelift_codegen::{
    entity::EntityRef,
    ir::{self, types, InstBuilder, MemFlags},
};
use cranelift_frontend::{FunctionBuilder, Variable};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};

use super::{Backend, FloatCmp, IntCmp, SemBlock, SemType, SemValue, SemVar};

pub(crate) struct CraneliftImports {
    pub(crate) str_cmp: FuncId,
    pub(crate) memcmp_eq: FuncId,
}

pub(crate) struct CraneliftBackend<'fb, 'ctx, 'm, 'i> {
    pub(crate) fb: &'fb mut FunctionBuilder<'ctx>,
    pub(crate) module: &'m mut JITModule,
    pub(crate) imports: &'i CraneliftImports,
}

impl CraneliftBackend<'_, '_, '_, '_> {
    fn ptr_type(&self) -> ir::Type {
        self.module.target_config().pointer_type()
    }

    fn to_ir_type(&self, ty: SemType) -> ir::Type {
        match ty {
            SemType::I8 => types::I8,
            SemType::I32 => types::I32,
            SemType::I64 => types::I64,
            SemType::F64 => types::F64,
            SemType::Ptr => self.ptr_type(),
        }
    }

    fn map_int_cmp(cc: IntCmp) -> ir::condcodes::IntCC {
        match cc {
            IntCmp::Eq => ir::condcodes::IntCC::Equal,
            IntCmp::NotEq => ir::condcodes::IntCC::NotEqual,
            IntCmp::SignedGt => ir::condcodes::IntCC::SignedGreaterThan,
            IntCmp::SignedGtEq => ir::condcodes::IntCC::SignedGreaterThanOrEqual,
            IntCmp::SignedLt => ir::condcodes::IntCC::SignedLessThan,
            IntCmp::SignedLtEq => ir::condcodes::IntCC::SignedLessThanOrEqual,
            IntCmp::UnsignedLtEq => ir::condcodes::IntCC::UnsignedLessThanOrEqual,
        }
    }

    fn map_float_cmp(cc: FloatCmp) -> ir::condcodes::FloatCC {
        match cc {
            FloatCmp::Eq => ir::condcodes::FloatCC::Equal,
            FloatCmp::NotEq => ir::condcodes::FloatCC::NotEqual,
            FloatCmp::Gt => ir::condcodes::FloatCC::GreaterThan,
            FloatCmp::GtEq => ir::condcodes::FloatCC::GreaterThanOrEqual,
            FloatCmp::Lt => ir::condcodes::FloatCC::LessThan,
            FloatCmp::LtEq => ir::condcodes::FloatCC::LessThanOrEqual,
        }
    }
}

impl Backend for CraneliftBackend<'_, '_, '_, '_> {
    fn pointer_bytes(&self) -> i64 {
        i64::from(self.ptr_type().bytes())
    }

    fn create_block(&mut self) -> SemBlock {
        SemBlock(self.fb.create_block())
    }

    fn append_block_params_for_function_params(&mut self, block: SemBlock) {
        self.fb.append_block_params_for_function_params(block.0);
    }

    fn append_block_param(&mut self, block: SemBlock, ty: SemType) {
        self.fb.append_block_param(block.0, self.to_ir_type(ty));
    }

    fn switch_to_block(&mut self, block: SemBlock) {
        self.fb.switch_to_block(block.0);
    }

    fn seal_block(&mut self, block: SemBlock) {
        self.fb.seal_block(block.0);
    }

    fn block_param(&mut self, block: SemBlock, index: usize) -> SemValue {
        SemValue(self.fb.block_params(block.0)[index])
    }

    fn jump(&mut self, block: SemBlock, args: &[SemValue]) {
        let vals = args.iter().map(|v| v.0).collect::<Vec<_>>();
        self.fb.ins().jump(block.0, &vals);
    }

    fn brif(
        &mut self,
        cond: SemValue,
        then_block: SemBlock,
        then_args: &[SemValue],
        else_block: SemBlock,
        else_args: &[SemValue],
    ) {
        let then_vals = then_args.iter().map(|v| v.0).collect::<Vec<_>>();
        let else_vals = else_args.iter().map(|v| v.0).collect::<Vec<_>>();
        self.fb
            .ins()
            .brif(cond.0, then_block.0, &then_vals, else_block.0, &else_vals);
    }

    fn declare_var(&mut self, var: SemVar, ty: SemType) {
        self.fb
            .declare_var(Variable::new(var.0), self.to_ir_type(ty));
    }

    fn def_var(&mut self, var: SemVar, value: SemValue) {
        self.fb.def_var(Variable::new(var.0), value.0);
    }

    fn use_var(&mut self, var: SemVar) -> SemValue {
        SemValue(self.fb.use_var(Variable::new(var.0)))
    }

    fn iconst_i64(&mut self, v: i64) -> SemValue {
        SemValue(self.fb.ins().iconst(types::I64, v))
    }

    fn iconst_i8(&mut self, v: i64) -> SemValue {
        SemValue(self.fb.ins().iconst(types::I8, v))
    }

    fn f64const(&mut self, v: f64) -> SemValue {
        SemValue(
            self.fb
                .ins()
                .f64const(ir::immediates::Ieee64::with_float(v)),
        )
    }

    fn iadd(&mut self, a: SemValue, b: SemValue) -> SemValue {
        SemValue(self.fb.ins().iadd(a.0, b.0))
    }

    fn iadd_imm(&mut self, a: SemValue, imm: i64) -> SemValue {
        SemValue(self.fb.ins().iadd_imm(a.0, imm))
    }

    fn isub(&mut self, a: SemValue, b: SemValue) -> SemValue {
        SemValue(self.fb.ins().isub(a.0, b.0))
    }

    fn imul_imm(&mut self, a: SemValue, imm: i64) -> SemValue {
        SemValue(self.fb.ins().imul_imm(a.0, imm))
    }

    fn icmp(&mut self, cc: IntCmp, a: SemValue, b: SemValue) -> SemValue {
        SemValue(self.fb.ins().icmp(Self::map_int_cmp(cc), a.0, b.0))
    }

    fn icmp_imm(&mut self, cc: IntCmp, a: SemValue, imm: i64) -> SemValue {
        SemValue(self.fb.ins().icmp_imm(Self::map_int_cmp(cc), a.0, imm))
    }

    fn fcmp(&mut self, cc: FloatCmp, a: SemValue, b: SemValue) -> SemValue {
        SemValue(self.fb.ins().fcmp(Self::map_float_cmp(cc), a.0, b.0))
    }

    fn band(&mut self, a: SemValue, b: SemValue) -> SemValue {
        SemValue(self.fb.ins().band(a.0, b.0))
    }

    fn bor(&mut self, a: SemValue, b: SemValue) -> SemValue {
        SemValue(self.fb.ins().bor(a.0, b.0))
    }

    fn bxor_imm(&mut self, a: SemValue, imm: i64) -> SemValue {
        SemValue(self.fb.ins().bxor_imm(a.0, imm))
    }

    fn select(&mut self, cond_b1: SemValue, yes: SemValue, no: SemValue) -> SemValue {
        SemValue(self.fb.ins().select(cond_b1.0, yes.0, no.0))
    }

    fn uextend(&mut self, to_ty: SemType, v: SemValue) -> SemValue {
        let ty = self.to_ir_type(to_ty);
        SemValue(self.fb.ins().uextend(ty, v.0))
    }

    fn load(&mut self, ty: SemType, base: SemValue, offset: i32) -> SemValue {
        let ty = self.to_ir_type(ty);
        SemValue(self.fb.ins().load(ty, MemFlags::new(), base.0, offset))
    }

    fn store(&mut self, value: SemValue, base: SemValue, offset: i32) {
        self.fb
            .ins()
            .store(MemFlags::new(), value.0, base.0, offset);
    }

    fn splat_i64x2(&mut self, scalar_i64: SemValue) -> SemValue {
        SemValue(self.fb.ins().splat(types::I64X2, scalar_i64.0))
    }

    fn load_i64x2(&mut self, base: SemValue, offset: i32) -> SemValue {
        SemValue(
            self.fb
                .ins()
                .load(types::I64X2, MemFlags::new(), base.0, offset),
        )
    }

    fn icmp_i64x2(&mut self, cc: IntCmp, lhs: SemValue, rhs: SemValue) -> SemValue {
        SemValue(self.fb.ins().icmp(Self::map_int_cmp(cc), lhs.0, rhs.0))
    }

    fn extract_lane(&mut self, vec: SemValue, lane: u8) -> SemValue {
        SemValue(self.fb.ins().extractlane(vec.0, lane))
    }

    fn call_str_cmp(
        &mut self,
        left_ptr: SemValue,
        left_len: SemValue,
        right_ptr: SemValue,
        right_len: SemValue,
        op_code: SemValue,
    ) -> SemValue {
        let fref = self
            .module
            .declare_func_in_func(self.imports.str_cmp, self.fb.func);
        let call = self.fb.ins().call(
            fref,
            &[left_ptr.0, left_len.0, right_ptr.0, right_len.0, op_code.0],
        );
        SemValue(self.fb.inst_results(call)[0])
    }

    fn call_memcmp_eq(
        &mut self,
        left_ptr: SemValue,
        right_ptr: SemValue,
        len: SemValue,
    ) -> SemValue {
        let fref = self
            .module
            .declare_func_in_func(self.imports.memcmp_eq, self.fb.func);
        let call = self.fb.ins().call(fref, &[left_ptr.0, right_ptr.0, len.0]);
        SemValue(self.fb.inst_results(call)[0])
    }

    fn return_void(&mut self) {
        self.fb.ins().return_(&[]);
    }
}

pub(crate) fn load_ptr_field<B: Backend>(backend: &mut B, base: SemValue, offset: i32) -> SemValue {
    backend.load(SemType::Ptr, base, offset)
}

pub(crate) fn load_table_ptr<B: Backend>(
    backend: &mut B,
    exec_ctx: SemValue,
    table_offset: i32,
    idx: i64,
) -> SemValue {
    let table = load_ptr_field(backend, exec_ctx, table_offset);
    let slot = backend.iadd_imm(table, idx * backend.pointer_bytes());
    backend.load(SemType::Ptr, slot, 0)
}

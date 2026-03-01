use arrow::datatypes::DataType;
use cranelift_codegen::ir::{types, AbiParam, Signature, UserFuncName};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module};

use super::{
    abi::{store_output_utf8view_passthrough, store_output_value},
    backend::{
        cranelift::{load_ptr_field, load_table_ptr, CraneliftBackend, CraneliftImports},
        Backend, IntCmp, SemBlock, SemType, SemValue, SemVar,
    },
    map_data_type,
    semantics::{
        expr::{SemanticCmp, SemanticExprBuilder},
        kernel::SemanticKernelBuilder,
        looping::{FilterStrategy, RowLoopSpec, SimdInt64FilterSpec},
    },
    CompileError,
};
use crate::{
    lowering::{FusedExpr, FusedLiteral, FusedProgram},
    runtime,
};

struct KernelBlocks {
    entry: SemBlock,
    loop_head: SemBlock,
    do_project: SemBlock,
    row_next: SemBlock,
    done: SemBlock,
}

struct KernelState {
    exec_ctx: SemValue,
    end: SemValue,
    row_var: SemVar,
    write_var: SemVar,
    write_idx_ptr: SemValue,
}

struct SemanticKernelCompiler<'a, B: Backend> {
    backend: &'a mut B,
    program: &'a FusedProgram,
    strategy: FilterStrategy,
    blocks: KernelBlocks,
    state: KernelState,
}

pub(super) fn compile_kernel_function(
    module: &mut JITModule,
    plan: &FusedProgram,
) -> Result<(FuncId, String), CompileError> {
    let strategy = detect_filter_strategy(plan.filter.as_ref());
    let semantic_kernel = SemanticKernelBuilder::new("jit_fused_filter_projection", plan)
        .row_loop(RowLoopSpec { step: 2 })
        .with_filter_strategy(strategy)
        .finalize();

    let (func_id, sig) = declare_kernel(module, semantic_kernel.name)?;
    let imports = declare_imports(module)?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    ctx.func.name = UserFuncName::user(0, func_id.as_u32());

    let mut fb_ctx = FunctionBuilderContext::new();
    let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

    {
        let mut backend = CraneliftBackend {
            fb: &mut fb,
            module,
            imports: &imports,
        };

        let blocks = KernelBlocks {
            entry: backend.create_block(),
            loop_head: backend.create_block(),
            do_project: backend.create_block(),
            row_next: backend.create_block(),
            done: backend.create_block(),
        };

        let state = emit_entry_setup(&mut backend, &blocks);
        let mut compiler = SemanticKernelCompiler {
            backend: &mut backend,
            program: semantic_kernel.plan,
            strategy: semantic_kernel.filter_strategy,
            blocks,
            state,
        };

        compiler.emit_loop(semantic_kernel.row_loop)?;
        compiler.emit_done();
        compiler.seal_blocks();
    }

    fb.finalize();
    let clif_ir = format!("{}", ctx.func.display());

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| CompileError::Internal(format!("define fused kernel failed: {e}")))?;
    module.clear_context(&mut ctx);
    Ok((func_id, clif_ir))
}

fn declare_kernel(
    module: &mut JITModule,
    function_name: &str,
) -> Result<(FuncId, Signature), CompileError> {
    let ptr = module.target_config().pointer_type();

    let mut sig = Signature::new(module.isa().default_call_conv());
    sig.params.push(AbiParam::new(ptr));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));

    let func_id = module
        .declare_function(function_name, Linkage::Local, &sig)
        .map_err(|e| CompileError::Internal(format!("declare fused kernel failed: {e}")))?;

    Ok((func_id, sig))
}

fn declare_imports(module: &mut JITModule) -> Result<CraneliftImports, CompileError> {
    let ptr = module.target_config().pointer_type();
    let mut sig = module.make_signature();
    sig.params.push(cranelift_codegen::ir::AbiParam::new(ptr));
    sig.params.push(cranelift_codegen::ir::AbiParam::new(
        cranelift_codegen::ir::types::I64,
    ));
    sig.params.push(cranelift_codegen::ir::AbiParam::new(ptr));
    sig.params.push(cranelift_codegen::ir::AbiParam::new(
        cranelift_codegen::ir::types::I64,
    ));
    sig.params.push(cranelift_codegen::ir::AbiParam::new(
        cranelift_codegen::ir::types::I64,
    ));
    sig.returns.push(cranelift_codegen::ir::AbiParam::new(
        cranelift_codegen::ir::types::I8,
    ));
    let str_cmp = module
        .declare_function("jit_str_cmp", Linkage::Import, &sig)
        .map_err(|e| CompileError::Internal(format!("declare jit_str_cmp failed: {e}")))?;

    let mut memcmp_sig = module.make_signature();
    memcmp_sig
        .params
        .push(cranelift_codegen::ir::AbiParam::new(ptr));
    memcmp_sig
        .params
        .push(cranelift_codegen::ir::AbiParam::new(ptr));
    memcmp_sig.params.push(cranelift_codegen::ir::AbiParam::new(
        cranelift_codegen::ir::types::I64,
    ));
    memcmp_sig
        .returns
        .push(cranelift_codegen::ir::AbiParam::new(
            cranelift_codegen::ir::types::I8,
        ));
    let memcmp_eq = module
        .declare_function("jit_memcmp_eq", Linkage::Import, &memcmp_sig)
        .map_err(|e| CompileError::Internal(format!("declare jit_memcmp_eq failed: {e}")))?;

    Ok(CraneliftImports { str_cmp, memcmp_eq })
}

fn emit_entry_setup<B: Backend>(backend: &mut B, blocks: &KernelBlocks) -> KernelState {
    backend.append_block_params_for_function_params(blocks.entry);
    backend.switch_to_block(blocks.entry);
    backend.seal_block(blocks.entry);

    let exec_ctx = backend.block_param(blocks.entry, 0);
    let start = backend.block_param(blocks.entry, 1);
    let len = backend.block_param(blocks.entry, 2);
    let end = backend.iadd(start, len);

    let row_var = SemVar(0);
    let write_var = SemVar(1);
    backend.declare_var(row_var, SemType::I64);
    backend.declare_var(write_var, SemType::I64);
    backend.def_var(row_var, start);

    let write_idx_ptr = load_ptr_field(backend, exec_ctx, runtime::OFFSET_WRITE_INDEX_PTR as i32);
    let init_write = backend.load(SemType::I64, write_idx_ptr, 0);
    backend.def_var(write_var, init_write);

    KernelState {
        exec_ctx,
        end,
        row_var,
        write_var,
        write_idx_ptr,
    }
}

impl<B: Backend> SemanticKernelCompiler<'_, B> {
    fn emit_loop(&mut self, row_loop: RowLoopSpec) -> Result<(), CompileError> {
        self.backend.jump(self.blocks.loop_head, &[]);

        self.backend.switch_to_block(self.blocks.loop_head);
        let row = self.backend.use_var(self.state.row_var);
        let done_cmp = self.backend.icmp(IntCmp::SignedGtEq, row, self.state.end);
        self.backend
            .brif(done_cmp, self.blocks.done, &[], self.blocks.do_project, &[]);

        self.backend.switch_to_block(self.blocks.do_project);
        self.emit_iteration_body(row)?;

        self.backend.switch_to_block(self.blocks.row_next);
        let next_row = self.backend.iadd_imm(row, row_loop.step);
        self.backend.def_var(self.state.row_var, next_row);
        self.backend.jump(self.blocks.loop_head, &[]);
        Ok(())
    }

    fn emit_iteration_body(&mut self, row: SemValue) -> Result<(), CompileError> {
        let row1 = self.backend.iadd_imm(row, 1);
        let lane1_active = self.backend.icmp(IntCmp::SignedLt, row1, self.state.end);

        match self.strategy {
            FilterStrategy::SimdInt64Literal(simd) => {
                self.emit_simd_iteration(simd, row, row1, lane1_active)
            }
            FilterStrategy::Scalar => self.emit_scalar_iteration(row, row1, lane1_active),
        }
    }

    fn emit_simd_iteration(
        &mut self,
        simd: SimdInt64FilterSpec,
        row: SemValue,
        row1: SemValue,
        lane1_active: SemValue,
    ) -> Result<(), CompileError> {
        let simd_eval = self.backend.create_block();
        let partial_eval = self.backend.create_block();
        self.backend
            .brif(lane1_active, simd_eval, &[], partial_eval, &[]);

        self.backend.switch_to_block(simd_eval);
        let (keep0, keep1) = self.emit_simd_int64_lane_keeps(simd, row);
        self.emit_conditional_projection(row, keep0)?;
        self.emit_conditional_projection(row1, keep1)?;
        self.backend.jump(self.blocks.row_next, &[]);

        self.backend.switch_to_block(partial_eval);
        let keep0 = self.emit_scalar_keep(row)?;
        self.emit_conditional_projection(row, keep0)?;
        self.backend.jump(self.blocks.row_next, &[]);

        self.backend.seal_block(simd_eval);
        self.backend.seal_block(partial_eval);
        Ok(())
    }

    fn emit_scalar_iteration(
        &mut self,
        row: SemValue,
        row1: SemValue,
        lane1_active: SemValue,
    ) -> Result<(), CompileError> {
        let keep0 = self.emit_scalar_keep(row)?;
        self.emit_conditional_projection(row, keep0)?;

        let lane1_block = self.backend.create_block();
        let merge_block = self.backend.create_block();
        self.backend
            .brif(lane1_active, lane1_block, &[], merge_block, &[]);

        self.backend.switch_to_block(lane1_block);
        let keep1 = self.emit_scalar_keep(row1)?;
        self.emit_conditional_projection(row1, keep1)?;
        self.backend.jump(merge_block, &[]);

        self.backend.switch_to_block(merge_block);
        self.backend.seal_block(lane1_block);
        self.backend.seal_block(merge_block);
        self.backend.jump(self.blocks.row_next, &[]);
        Ok(())
    }

    fn emit_simd_int64_lane_keeps(
        &mut self,
        simd: SimdInt64FilterSpec,
        row: SemValue,
    ) -> (SemValue, SemValue) {
        let in_ptr = load_table_ptr(
            self.backend,
            self.state.exec_ctx,
            runtime::OFFSET_INPUT_PTRS as i32,
            simd.col_idx,
        );
        let byte_off = self.backend.imul_imm(row, 8);
        let addr = self.backend.iadd(in_ptr, byte_off);
        let vals = self.backend.load_i64x2(addr, 0);
        let lit_scalar = self.backend.iconst_i64(simd.lit);
        let lit_vec = self.backend.splat_i64x2(lit_scalar);
        let cmp_vec = self
            .backend
            .icmp_i64x2(map_semantic_cmp_to_int(simd.cmp), vals, lit_vec);

        let lane0 = self.backend.extract_lane(cmp_vec, 0);
        let lane1 = self.backend.extract_lane(cmp_vec, 1);
        let keep0 = self.backend.icmp_imm(IntCmp::NotEq, lane0, 0);
        let keep1 = self.backend.icmp_imm(IntCmp::NotEq, lane1, 0);
        (keep0, keep1)
    }

    fn emit_scalar_keep(&mut self, row: SemValue) -> Result<SemValue, CompileError> {
        if let Some(predicate) = self.program.filter.as_ref() {
            let keep = {
                let mut expr = SemanticExprBuilder {
                    backend: self.backend,
                    exec_ctx: self.state.exec_ctx,
                };
                expr.emit_expr(row, predicate)?.to_bool_i8()
            };
            Ok(self.backend.icmp_imm(IntCmp::NotEq, keep, 0))
        } else {
            Ok(self.backend.icmp(IntCmp::Eq, row, row))
        }
    }

    fn emit_conditional_projection(
        &mut self,
        row: SemValue,
        keep_cmp: SemValue,
    ) -> Result<(), CompileError> {
        let project_block = self.backend.create_block();
        let merge_block = self.backend.create_block();
        self.backend
            .brif(keep_cmp, project_block, &[], merge_block, &[]);

        self.backend.switch_to_block(project_block);
        let write_idx = self.backend.use_var(self.state.write_var);

        for (i, expr) in self.program.projection.iter().enumerate() {
            if let FusedExpr::Column(col) = expr {
                let out_dt = self.program.output_schema.field(i).data_type();
                if matches!(out_dt, DataType::Utf8View)
                    && matches!(col.data_type, DataType::Utf8View)
                {
                    store_output_utf8view_passthrough(
                        self.backend,
                        self.state.exec_ctx,
                        i as i64,
                        col.index as i64,
                        row,
                        write_idx,
                    );
                    continue;
                }
            }

            let value = {
                let mut expr_builder = SemanticExprBuilder {
                    backend: self.backend,
                    exec_ctx: self.state.exec_ctx,
                };
                expr_builder.emit_expr(row, expr)?
            };
            let output_type = map_data_type(self.program.output_schema.field(i).data_type())?;
            store_output_value(
                self.backend,
                self.state.exec_ctx,
                i as i64,
                write_idx,
                value,
                output_type,
            )?;
        }

        let next_write = self.backend.iadd_imm(write_idx, 1);
        self.backend.def_var(self.state.write_var, next_write);
        self.backend.jump(merge_block, &[]);

        self.backend.switch_to_block(merge_block);
        self.backend.seal_block(project_block);
        self.backend.seal_block(merge_block);
        Ok(())
    }

    fn emit_done(&mut self) {
        self.backend.switch_to_block(self.blocks.done);
        let final_write = self.backend.use_var(self.state.write_var);
        self.backend.store(final_write, self.state.write_idx_ptr, 0);
        self.backend.return_void();
    }

    fn seal_blocks(&mut self) {
        self.backend.seal_block(self.blocks.loop_head);
        self.backend.seal_block(self.blocks.do_project);
        self.backend.seal_block(self.blocks.row_next);
        self.backend.seal_block(self.blocks.done);
    }
}

fn detect_filter_strategy(filter: Option<&FusedExpr>) -> FilterStrategy {
    let binary = match filter {
        Some(FusedExpr::Binary(b)) => b,
        _ => return FilterStrategy::Scalar,
    };

    let cmp = match binary.op {
        query_core::logical_exprs::binary::Operator::Gt => SemanticCmp::Gt,
        query_core::logical_exprs::binary::Operator::GtEq => SemanticCmp::GtEq,
        query_core::logical_exprs::binary::Operator::Lt => SemanticCmp::Lt,
        query_core::logical_exprs::binary::Operator::LtEq => SemanticCmp::LtEq,
        query_core::logical_exprs::binary::Operator::Eq => SemanticCmp::Eq,
        query_core::logical_exprs::binary::Operator::NotEq => SemanticCmp::NotEq,
        query_core::logical_exprs::binary::Operator::And
        | query_core::logical_exprs::binary::Operator::Or => return FilterStrategy::Scalar,
    };

    match (binary.left.as_ref(), binary.right.as_ref()) {
        (FusedExpr::Column(col), FusedExpr::Literal(FusedLiteral::Long(lit)))
            if matches!(col.data_type, DataType::Int64) =>
        {
            FilterStrategy::SimdInt64Literal(SimdInt64FilterSpec {
                col_idx: col.index as i64,
                lit: *lit,
                cmp,
            })
        }
        _ => FilterStrategy::Scalar,
    }
}

fn map_semantic_cmp_to_int(cmp: SemanticCmp) -> IntCmp {
    match cmp {
        SemanticCmp::Eq => IntCmp::Eq,
        SemanticCmp::NotEq => IntCmp::NotEq,
        SemanticCmp::Gt => IntCmp::SignedGt,
        SemanticCmp::GtEq => IntCmp::SignedGtEq,
        SemanticCmp::Lt => IntCmp::SignedLt,
        SemanticCmp::LtEq => IntCmp::SignedLtEq,
    }
}

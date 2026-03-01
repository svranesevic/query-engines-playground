use std::ffi::c_void;

use arrow::datatypes::DataType;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::default_libcall_names;

use super::{lowering::FusedProgram, runtime, CompileError};

mod abi;
mod backend;
mod expr_codegen;
mod kernel_codegen;
mod semantics;

/// JIT-compiled fused stage entrypoint ABI.
pub type FusedKernel = unsafe extern "C" fn(*mut c_void, i64, i64);

pub struct CraneliftFusedKernel {
    _module: JITModule,
    pub func: FusedKernel,
    pub clif_ir: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum JitType {
    I64,
    U64,
    F64,
    Bool,
    Utf8,
    Utf8View,
}

pub fn compile_fused_kernel(plan: &FusedProgram) -> Result<CraneliftFusedKernel, CompileError> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| CompileError::Internal(format!("failed to set cranelift opt level: {e}")))?;

    let isa_builder = cranelift_native::builder()
        .map_err(|e| CompileError::Internal(format!("failed to create native ISA builder: {e}")))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| CompileError::Internal(format!("failed to build ISA: {e}")))?;

    let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
    builder.symbol("jit_str_cmp", runtime::jit_str_cmp as *const u8);
    builder.symbol("jit_memcmp_eq", runtime::jit_memcmp_eq as *const u8);

    let mut module = JITModule::new(builder);

    let (func_id, clif_ir) = kernel_codegen::compile_kernel_function(&mut module, plan)?;

    module
        .finalize_definitions()
        .map_err(|e| CompileError::Internal(format!("failed to finalize fused kernel: {e}")))?;

    let ptr = module.get_finalized_function(func_id);
    let func = unsafe { std::mem::transmute::<*const u8, FusedKernel>(ptr) };

    Ok(CraneliftFusedKernel {
        _module: module,
        func,
        clif_ir,
    })
}

pub(super) fn map_data_type(dt: &DataType) -> Result<JitType, CompileError> {
    match dt {
        DataType::Int64 => Ok(JitType::I64),
        DataType::UInt64 => Ok(JitType::U64),
        DataType::Float64 => Ok(JitType::F64),
        DataType::Boolean => Ok(JitType::Bool),
        DataType::Utf8 => Ok(JitType::Utf8),
        DataType::Utf8View => Ok(JitType::Utf8View),
        other => Err(CompileError::UnsupportedType(format!(
            "unsupported type in codegen: {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};

    use crate::lowering::{
        FusedBinaryExpr, FusedColumnExpr, FusedExpr, FusedLiteral, FusedProgram,
    };
    use query_core::logical_exprs::binary::Operator;

    use super::compile_fused_kernel;

    #[test]
    fn clif_contains_simd_ops_for_int64_col_lit_filter() {
        let schema = Schema::new(vec![Field::new("age", DataType::Int64, false)]);
        let program = FusedProgram {
            filter: Some(FusedExpr::Binary(FusedBinaryExpr {
                left: Box::new(FusedExpr::Column(FusedColumnExpr {
                    index: 0,
                    data_type: DataType::Int64,
                })),
                right: Box::new(FusedExpr::Literal(FusedLiteral::Long(20))),
                op: Operator::Gt,
                data_type: DataType::Boolean,
            })),
            projection: vec![FusedExpr::Column(FusedColumnExpr {
                index: 0,
                data_type: DataType::Int64,
            })],
            output_schema: schema,
            string_literals: vec![],
        };

        let kernel = compile_fused_kernel(&program).unwrap();
        assert!(kernel.clif_ir.contains("extractlane"));
        assert!(kernel.clif_ir.contains("splat"));
    }

    #[test]
    fn clif_has_row_loop_shape() {
        let schema = Schema::new(vec![Field::new("age", DataType::Int64, false)]);
        let program = FusedProgram {
            filter: None,
            projection: vec![FusedExpr::Column(FusedColumnExpr {
                index: 0,
                data_type: DataType::Int64,
            })],
            output_schema: schema,
            string_literals: vec![],
        };

        let kernel = compile_fused_kernel(&program).unwrap();
        assert!(kernel.clif_ir.contains("brif"));
        assert!(kernel.clif_ir.contains("iadd_imm"));
    }
}

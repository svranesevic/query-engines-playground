use crate::{codegen::expr_codegen::SemGenValue, runtime, CompileError};

use super::{
    backend::{Backend, SemType, SemValue},
    JitType,
};

pub(super) fn load_ptr_field<B: Backend>(backend: &mut B, base: SemValue, offset: i32) -> SemValue {
    backend.load(SemType::Ptr, base, offset)
}

pub(super) fn load_table_ptr<B: Backend>(
    backend: &mut B,
    exec_ctx: SemValue,
    table_offset: i32,
    idx: i64,
) -> SemValue {
    let table = load_ptr_field(backend, exec_ctx, table_offset);
    let slot = backend.iadd_imm(table, idx * backend.pointer_bytes());
    backend.load(SemType::Ptr, slot, 0)
}

pub(super) fn store_output_value<B: Backend>(
    backend: &mut B,
    exec_ctx: SemValue,
    proj_idx: i64,
    write_idx: SemValue,
    value: SemGenValue,
    ty: JitType,
) -> Result<(), CompileError> {
    let out_ptr = load_table_ptr(
        backend,
        exec_ctx,
        runtime::OFFSET_OUTPUT_PTRS as i32,
        proj_idx,
    );

    match (value, ty) {
        (SemGenValue::I64(v), JitType::I64) => {
            let byte_off = backend.imul_imm(write_idx, 8);
            let addr = backend.iadd(out_ptr, byte_off);
            backend.store(v, addr, 0);
        }
        (SemGenValue::U64(v), JitType::U64) => {
            let byte_off = backend.imul_imm(write_idx, 8);
            let addr = backend.iadd(out_ptr, byte_off);
            backend.store(v, addr, 0);
        }
        (SemGenValue::F64(v), JitType::F64) => {
            let byte_off = backend.imul_imm(write_idx, 8);
            let addr = backend.iadd(out_ptr, byte_off);
            backend.store(v, addr, 0);
        }
        (SemGenValue::BoolI8(v), JitType::Bool) => {
            let addr = backend.iadd(out_ptr, write_idx);
            backend.store(v, addr, 0);
        }
        (SemGenValue::Str { ptr, len }, JitType::Utf8 | JitType::Utf8View) => {
            let aux_ptr = load_table_ptr(
                backend,
                exec_ctx,
                runtime::OFFSET_OUTPUT_AUX_PTRS as i32,
                proj_idx,
            );
            let byte_off = backend.imul_imm(write_idx, 8);
            let ptr_addr = backend.iadd(out_ptr, byte_off);
            let len_addr = backend.iadd(aux_ptr, byte_off);
            backend.store(ptr, ptr_addr, 0);
            backend.store(len, len_addr, 0);
        }
        _ => {
            return Err(CompileError::Internal(format!(
                "projection output value/type mismatch in codegen: value={value:?}, ty={ty:?}"
            )));
        }
    }
    Ok(())
}

pub(super) fn store_output_utf8view_passthrough<B: Backend>(
    backend: &mut B,
    exec_ctx: SemValue,
    proj_idx: i64,
    input_col_idx: i64,
    row: SemValue,
    write_idx: SemValue,
) {
    let in_ptr = load_table_ptr(
        backend,
        exec_ctx,
        runtime::OFFSET_INPUT_PTRS as i32,
        input_col_idx,
    );
    let out_ptr = load_table_ptr(
        backend,
        exec_ctx,
        runtime::OFFSET_OUTPUT_PTRS as i32,
        proj_idx,
    );
    let in_off = backend.imul_imm(row, 16);
    let out_off = backend.imul_imm(write_idx, 16);
    let in_addr = backend.iadd(in_ptr, in_off);
    let out_addr = backend.iadd(out_ptr, out_off);

    let lo = backend.load(SemType::I64, in_addr, 0);
    let hi = backend.load(SemType::I64, in_addr, 8);
    backend.store(lo, out_addr, 0);
    backend.store(hi, out_addr, 8);
}

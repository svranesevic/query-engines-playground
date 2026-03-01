use arrow::datatypes::DataType;

use crate::{
    codegen::{expr_codegen::SemGenValue, JitType},
    lowering::{FusedExpr, FusedLiteral},
    runtime, CompileError,
};
use query_core::logical_exprs::binary::Operator;

use super::super::{
    backend::{Backend, FloatCmp, IntCmp, SemType, SemValue},
    map_data_type,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SemanticCmp {
    Eq,
    NotEq,
    Gt,
    GtEq,
    Lt,
    LtEq,
}

pub(crate) struct SemanticExprBuilder<'a, B: Backend> {
    pub(crate) backend: &'a mut B,
    pub(crate) exec_ctx: SemValue,
}

impl<'a, B: Backend> SemanticExprBuilder<'a, B> {
    pub(crate) fn emit_expr(
        &mut self,
        row: SemValue,
        expr: &FusedExpr,
    ) -> Result<SemGenValue, CompileError> {
        match expr {
            FusedExpr::Column(col) => {
                let ty = map_data_type(&col.data_type)?;
                Ok(self.load_input_value(row, col.index as i64, ty))
            }
            FusedExpr::Literal(lit) => Ok(self.emit_literal(lit)),
            FusedExpr::Binary(binary) => {
                if matches!(binary.op, Operator::Eq | Operator::NotEq) {
                    if let (FusedExpr::Column(left_col), FusedExpr::Column(right_col)) =
                        (binary.left.as_ref(), binary.right.as_ref())
                    {
                        if matches!(left_col.data_type, DataType::Utf8View)
                            && matches!(right_col.data_type, DataType::Utf8View)
                        {
                            let v = self.emit_utf8view_col_eq_neq(
                                row,
                                left_col.index as i64,
                                right_col.index as i64,
                                matches!(binary.op, Operator::NotEq),
                            );
                            return Ok(SemGenValue::BoolI8(v));
                        }
                    }
                }

                let left = self.emit_expr(row, binary.left.as_ref())?;
                let right = self.emit_expr(row, binary.right.as_ref())?;

                match &binary.op {
                    Operator::And => {
                        let l = left.to_bool_i8();
                        let r = right.to_bool_i8();
                        Ok(SemGenValue::BoolI8(self.backend.band(l, r)))
                    }
                    Operator::Or => {
                        let l = left.to_bool_i8();
                        let r = right.to_bool_i8();
                        Ok(SemGenValue::BoolI8(self.backend.bor(l, r)))
                    }
                    op => {
                        let b1 = match (left, right) {
                            (SemGenValue::I64(l), SemGenValue::I64(r)) => {
                                self.backend.icmp(map_intcc_signed(op), l, r)
                            }
                            (SemGenValue::U64(l), SemGenValue::U64(r)) => {
                                self.backend.icmp(map_intcc_unsigned(op), l, r)
                            }
                            (SemGenValue::F64(l), SemGenValue::F64(r)) => {
                                self.backend.fcmp(map_floatcc(op), l, r)
                            }
                            (SemGenValue::BoolI8(l), SemGenValue::BoolI8(r)) => {
                                let cc = match op {
                                    Operator::Eq => IntCmp::Eq,
                                    Operator::NotEq => IntCmp::NotEq,
                                    _ => {
                                        return Err(CompileError::UnsupportedExpr(format!(
                                            "unsupported bool comparator {:?}",
                                            op
                                        )));
                                    }
                                };
                                self.backend.icmp(cc, l, r)
                            }
                            (
                                SemGenValue::Str { ptr: lp, len: ll },
                                SemGenValue::Str { ptr: rp, len: rl },
                            ) => {
                                if matches!(*op, Operator::Eq | Operator::NotEq) {
                                    return Ok(SemGenValue::BoolI8(self.emit_string_eq_neq(
                                        lp,
                                        ll,
                                        rp,
                                        rl,
                                        matches!(*op, Operator::NotEq),
                                    )));
                                }
                                let op_code = self.backend.iconst_i64(map_op_code(op));
                                return Ok(SemGenValue::BoolI8(
                                    self.backend.call_str_cmp(lp, ll, rp, rl, op_code),
                                ));
                            }
                            _ => {
                                return Err(CompileError::UnsupportedExpr(
                                    "binary operand kind mismatch in codegen".to_string(),
                                ));
                            }
                        };
                        Ok(SemGenValue::BoolI8(self.bool_to_i8(b1)))
                    }
                }
            }
        }
    }

    fn emit_literal(&mut self, lit: &FusedLiteral) -> SemGenValue {
        match lit {
            FusedLiteral::Long(v) => SemGenValue::I64(self.backend.iconst_i64(*v)),
            FusedLiteral::UInt64(v) => SemGenValue::U64(self.backend.iconst_i64(*v as i64)),
            FusedLiteral::Double(v) => SemGenValue::F64(self.backend.f64const(*v)),
            FusedLiteral::Str { literal_idx } => {
                let idx = *literal_idx as i64;
                let lit_ptr = self.load_table_ptr(runtime::OFFSET_LITERAL_PTRS as i32, idx);
                let lit_len = self.load_table_ptr(runtime::OFFSET_LITERAL_LENS as i32, idx);
                SemGenValue::Str {
                    ptr: lit_ptr,
                    len: lit_len,
                }
            }
        }
    }

    fn bool_to_i8(&mut self, b1: SemValue) -> SemValue {
        let one = self.backend.iconst_i8(1);
        let zero = self.backend.iconst_i8(0);
        self.backend.select(b1, one, zero)
    }

    fn emit_string_eq_neq(
        &mut self,
        left_ptr: SemValue,
        left_len: SemValue,
        right_ptr: SemValue,
        right_len: SemValue,
        negate: bool,
    ) -> SemValue {
        let len_eq = self.backend.icmp(IntCmp::Eq, left_len, right_len);

        let eq_len_block = self.backend.create_block();
        let merge_block = self.backend.create_block();
        self.backend.append_block_param(merge_block, SemType::I8);

        let mismatch = self.backend.iconst_i8(i64::from(negate));
        self.backend
            .brif(len_eq, eq_len_block, &[], merge_block, &[mismatch]);

        self.backend.switch_to_block(eq_len_block);
        let eq = self.backend.call_memcmp_eq(left_ptr, right_ptr, left_len);
        let out = if negate {
            self.backend.bxor_imm(eq, 1)
        } else {
            eq
        };
        self.backend.jump(merge_block, &[out]);

        self.backend.switch_to_block(merge_block);
        self.backend.seal_block(eq_len_block);
        self.backend.seal_block(merge_block);
        self.backend.block_param(merge_block, 0)
    }

    fn emit_utf8view_col_eq_neq(
        &mut self,
        row: SemValue,
        left_col_idx: i64,
        right_col_idx: i64,
        negate: bool,
    ) -> SemValue {
        let left_views = self.load_table_ptr(runtime::OFFSET_INPUT_PTRS as i32, left_col_idx);
        let right_views = self.load_table_ptr(runtime::OFFSET_INPUT_PTRS as i32, right_col_idx);
        let left_buf_table =
            self.load_table_ptr(runtime::OFFSET_INPUT_AUX_PTRS as i32, left_col_idx);
        let right_buf_table =
            self.load_table_ptr(runtime::OFFSET_INPUT_AUX_PTRS as i32, right_col_idx);

        let row_off = self.backend.imul_imm(row, 16);
        let left_view_addr = self.backend.iadd(left_views, row_off);
        let right_view_addr = self.backend.iadd(right_views, row_off);

        let left_len = self.backend.load(SemType::I32, left_view_addr, 0);
        let right_len = self.backend.load(SemType::I32, right_view_addr, 0);

        let len_eq = self.backend.icmp(IntCmp::Eq, left_len, right_len);

        let len_match_block = self.backend.create_block();
        let short_block = self.backend.create_block();
        let long_block = self.backend.create_block();
        let merge_block = self.backend.create_block();
        self.backend.append_block_param(merge_block, SemType::I8);

        let mismatch = self.backend.iconst_i8(i64::from(negate));
        self.backend
            .brif(len_eq, len_match_block, &[], merge_block, &[mismatch]);

        self.backend.switch_to_block(len_match_block);
        let short_cmp = self.backend.icmp_imm(IntCmp::UnsignedLtEq, left_len, 12);
        self.backend
            .brif(short_cmp, short_block, &[], long_block, &[]);

        self.backend.switch_to_block(short_block);
        let l0 = self.backend.load(SemType::I64, left_view_addr, 0);
        let l1 = self.backend.load(SemType::I64, left_view_addr, 8);
        let r0 = self.backend.load(SemType::I64, right_view_addr, 0);
        let r1 = self.backend.load(SemType::I64, right_view_addr, 8);
        let low_eq = self.backend.icmp(IntCmp::Eq, l0, r0);
        let high_eq = self.backend.icmp(IntCmp::Eq, l1, r1);
        let both = self.backend.band(low_eq, high_eq);
        let mut eq_short = self.bool_to_i8(both);
        if negate {
            eq_short = self.backend.bxor_imm(eq_short, 1);
        }
        self.backend.jump(merge_block, &[eq_short]);

        self.backend.switch_to_block(long_block);
        let l_prefix = self.backend.load(SemType::I32, left_view_addr, 4);
        let r_prefix = self.backend.load(SemType::I32, right_view_addr, 4);
        let prefix_eq = self.backend.icmp(IntCmp::Eq, l_prefix, r_prefix);

        let prefix_match_block = self.backend.create_block();
        self.backend
            .brif(prefix_eq, prefix_match_block, &[], merge_block, &[mismatch]);

        self.backend.switch_to_block(prefix_match_block);
        let l_buf_i32 = self.backend.load(SemType::I32, left_view_addr, 8);
        let r_buf_i32 = self.backend.load(SemType::I32, right_view_addr, 8);
        let l_off_i32 = self.backend.load(SemType::I32, left_view_addr, 12);
        let r_off_i32 = self.backend.load(SemType::I32, right_view_addr, 12);

        let l_buf_i64 = self.backend.uextend(SemType::I64, l_buf_i32);
        let r_buf_i64 = self.backend.uextend(SemType::I64, r_buf_i32);
        let l_off_i64 = self.backend.uextend(SemType::I64, l_off_i32);
        let r_off_i64 = self.backend.uextend(SemType::I64, r_off_i32);

        let l_slot_off = self
            .backend
            .imul_imm(l_buf_i64, self.backend.pointer_bytes());
        let r_slot_off = self
            .backend
            .imul_imm(r_buf_i64, self.backend.pointer_bytes());
        let l_slot = self.backend.iadd(left_buf_table, l_slot_off);
        let r_slot = self.backend.iadd(right_buf_table, r_slot_off);
        let l_data = self.backend.load(SemType::Ptr, l_slot, 0);
        let r_data = self.backend.load(SemType::Ptr, r_slot, 0);
        let l_ptr = self.backend.iadd(l_data, l_off_i64);
        let r_ptr = self.backend.iadd(r_data, r_off_i64);
        let len_i64 = self.backend.uextend(SemType::I64, left_len);

        let mut eq_long = self.backend.call_memcmp_eq(l_ptr, r_ptr, len_i64);
        if negate {
            eq_long = self.backend.bxor_imm(eq_long, 1);
        }
        self.backend.jump(merge_block, &[eq_long]);

        self.backend.switch_to_block(merge_block);
        self.backend.seal_block(len_match_block);
        self.backend.seal_block(short_block);
        self.backend.seal_block(long_block);
        self.backend.seal_block(prefix_match_block);
        self.backend.seal_block(merge_block);

        self.backend.block_param(merge_block, 0)
    }

    fn load_ptr_field(&mut self, base: SemValue, offset: i32) -> SemValue {
        self.backend.load(SemType::Ptr, base, offset)
    }

    fn load_table_ptr(&mut self, table_offset: i32, idx: i64) -> SemValue {
        let table = self.load_ptr_field(self.exec_ctx, table_offset);
        let slot = self
            .backend
            .iadd_imm(table, idx * self.backend.pointer_bytes());
        self.backend.load(SemType::Ptr, slot, 0)
    }

    fn load_input_value(&mut self, row: SemValue, col_idx: i64, ty: JitType) -> SemGenValue {
        let col_ptr = self.load_table_ptr(runtime::OFFSET_INPUT_PTRS as i32, col_idx);
        match ty {
            JitType::I64 => {
                let byte_off = self.backend.imul_imm(row, 8);
                let addr = self.backend.iadd(col_ptr, byte_off);
                SemGenValue::I64(self.backend.load(SemType::I64, addr, 0))
            }
            JitType::U64 => {
                let byte_off = self.backend.imul_imm(row, 8);
                let addr = self.backend.iadd(col_ptr, byte_off);
                SemGenValue::U64(self.backend.load(SemType::I64, addr, 0))
            }
            JitType::F64 => {
                let byte_off = self.backend.imul_imm(row, 8);
                let addr = self.backend.iadd(col_ptr, byte_off);
                SemGenValue::F64(self.backend.load(SemType::F64, addr, 0))
            }
            JitType::Bool => {
                let addr = self.backend.iadd(col_ptr, row);
                SemGenValue::BoolI8(self.backend.load(SemType::I8, addr, 0))
            }
            JitType::Utf8 => {
                let data_ptr = self.load_table_ptr(runtime::OFFSET_INPUT_AUX_PTRS as i32, col_idx);
                let row_off = self.backend.imul_imm(row, 4);
                let start_addr = self.backend.iadd(col_ptr, row_off);
                let end_addr = self.backend.iadd_imm(start_addr, 4);
                let start = self.backend.load(SemType::I32, start_addr, 0);
                let end = self.backend.load(SemType::I32, end_addr, 0);
                let len_i32 = self.backend.isub(end, start);
                let start_i64 = self.backend.uextend(SemType::I64, start);
                let len_i64 = self.backend.uextend(SemType::I64, len_i32);
                let ptr = self.backend.iadd(data_ptr, start_i64);
                SemGenValue::Str { ptr, len: len_i64 }
            }
            JitType::Utf8View => {
                let len_ptr = self.load_table_ptr(runtime::OFFSET_INPUT_AUX_PTRS as i32, col_idx);
                let row_off = self.backend.imul_imm(row, 16);
                let view_addr = self.backend.iadd(col_ptr, row_off);

                let len_i32 = self.backend.load(SemType::I32, view_addr, 0);
                let len_i64 = self.backend.uextend(SemType::I64, len_i32);
                let is_inline = self.backend.icmp_imm(IntCmp::UnsignedLtEq, len_i32, 12);

                let inline_block = self.backend.create_block();
                let external_block = self.backend.create_block();
                let merge_block = self.backend.create_block();
                self.backend.append_block_param(merge_block, SemType::Ptr);
                self.backend.append_block_param(merge_block, SemType::I64);

                self.backend
                    .brif(is_inline, inline_block, &[], external_block, &[]);

                self.backend.switch_to_block(inline_block);
                let inline_ptr = self.backend.iadd_imm(view_addr, 4);
                self.backend.jump(merge_block, &[inline_ptr, len_i64]);

                self.backend.switch_to_block(external_block);
                let buffer_index_addr = self.backend.iadd_imm(view_addr, 8);
                let offset_addr = self.backend.iadd_imm(view_addr, 12);
                let buffer_index_i32 = self.backend.load(SemType::I32, buffer_index_addr, 0);
                let offset_i32 = self.backend.load(SemType::I32, offset_addr, 0);
                let buffer_index_i64 = self.backend.uextend(SemType::I64, buffer_index_i32);
                let offset_i64 = self.backend.uextend(SemType::I64, offset_i32);

                let table_off = self
                    .backend
                    .imul_imm(buffer_index_i64, self.backend.pointer_bytes());
                let table_slot = self.backend.iadd(len_ptr, table_off);
                let data_buf_ptr = self.backend.load(SemType::Ptr, table_slot, 0);
                let external_ptr = self.backend.iadd(data_buf_ptr, offset_i64);
                self.backend.jump(merge_block, &[external_ptr, len_i64]);

                self.backend.switch_to_block(merge_block);
                self.backend.seal_block(inline_block);
                self.backend.seal_block(external_block);
                self.backend.seal_block(merge_block);

                let p = self.backend.block_param(merge_block, 0);
                let l = self.backend.block_param(merge_block, 1);
                SemGenValue::Str { ptr: p, len: l }
            }
        }
    }
}

fn map_intcc_signed(op: &Operator) -> IntCmp {
    match op {
        Operator::Gt => IntCmp::SignedGt,
        Operator::GtEq => IntCmp::SignedGtEq,
        Operator::Lt => IntCmp::SignedLt,
        Operator::LtEq => IntCmp::SignedLtEq,
        Operator::Eq => IntCmp::Eq,
        Operator::NotEq => IntCmp::NotEq,
        Operator::And | Operator::Or => panic!("invalid int comparator"),
    }
}

fn map_intcc_unsigned(op: &Operator) -> IntCmp {
    match op {
        Operator::Gt => IntCmp::SignedGt,
        Operator::GtEq => IntCmp::SignedGtEq,
        Operator::Lt => IntCmp::SignedLt,
        Operator::LtEq => IntCmp::SignedLtEq,
        Operator::Eq => IntCmp::Eq,
        Operator::NotEq => IntCmp::NotEq,
        Operator::And | Operator::Or => panic!("invalid uint comparator"),
    }
}

fn map_floatcc(op: &Operator) -> FloatCmp {
    match op {
        Operator::Gt => FloatCmp::Gt,
        Operator::GtEq => FloatCmp::GtEq,
        Operator::Lt => FloatCmp::Lt,
        Operator::LtEq => FloatCmp::LtEq,
        Operator::Eq => FloatCmp::Eq,
        Operator::NotEq => FloatCmp::NotEq,
        Operator::And | Operator::Or => panic!("invalid float comparator"),
    }
}

fn map_op_code(op: &Operator) -> i64 {
    match op {
        Operator::Gt => 0,
        Operator::GtEq => 1,
        Operator::Lt => 2,
        Operator::LtEq => 3,
        Operator::Eq => 4,
        Operator::NotEq => 5,
        Operator::And | Operator::Or => panic!("logical op not valid for string compare"),
    }
}

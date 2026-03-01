use std::{collections::HashMap, ffi::c_void, ptr, sync::Arc};

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
    StringViewArray, UInt64Array,
};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Schema};

use volcano_engine::physical_exprs::{aggregate::AggregateExpr, PhysicalExpr};

use super::{lowering::FusedProgram, CompileError};

pub const OFFSET_INPUT_PTRS: usize = 0;
pub const OFFSET_INPUT_AUX_PTRS: usize = 8;
pub const OFFSET_OUTPUT_PTRS: usize = 16;
pub const OFFSET_OUTPUT_AUX_PTRS: usize = 24;
pub const OFFSET_LITERAL_PTRS: usize = 32;
pub const OFFSET_LITERAL_LENS: usize = 40;
pub const OFFSET_WRITE_INDEX_PTR: usize = 48;

#[repr(C)]
/// FFI context shared between Rust runtime and generated JIT kernel.
///
/// The runtime fills this with pointer tables for input/output buffers and
/// literal storage before invoking the compiled kernel. Generated code treats
/// it as a stable ABI contract and reads fields by fixed offsets.
pub struct ExecCtx {
    /// Table of input "primary" pointers, one entry per input column.
    /// Interpretation depends on input type:
    /// - Int64/UInt64/Float64: values buffer pointer
    /// - Boolean: byte-per-row buffer pointer (0/1)
    /// - Utf8: i32 offsets buffer pointer
    /// - Utf8View: u128 views buffer pointer
    pub input_ptrs: *const *const u8,
    /// Table of input auxiliary pointers, one entry per input column.
    /// Interpretation depends on input type:
    /// - Utf8: string data buffer pointer
    /// - Utf8View: table pointer to underlying data buffer pointers
    /// - other types: null
    pub input_aux_ptrs: *const *const u8,
    /// Table of output "primary" pointers, one entry per projected output column.
    pub output_ptrs: *const *mut u8,
    /// Table of output auxiliary pointers, one entry per projected output column.
    pub output_aux_ptrs: *const *mut u8,
    /// Pointers to interned string literal bytes referenced by compiled expressions.
    pub literal_ptrs: *const *const u8,
    /// Lengths for `literal_ptrs` entries.
    pub literal_lens: *const usize,
    /// Mutable write index incremented by the kernel when a row is emitted.
    pub write_index_ptr: *mut i64,
}

enum InputBacking {
    BoolBytes { _bytes: Vec<u8> },
    ViewBufferPtrs { _ptrs: Vec<usize> },
}

enum OutputStorage {
    I64(Vec<i64>),
    U64(Vec<u64>),
    F64(Vec<f64>),
    Bool(Vec<u8>),
    Str {
        ptrs: Vec<usize>,
        lens: Vec<usize>,
        view: bool,
    },
    ViewRaw {
        views: Vec<u128>,
        buffers: Vec<Buffer>,
    },
}

impl OutputStorage {
    fn new(dtype: &DataType, capacity: usize) -> Result<Self, CompileError> {
        Ok(match dtype {
            DataType::Int64 => OutputStorage::I64(vec![0; capacity]),
            DataType::UInt64 => OutputStorage::U64(vec![0; capacity]),
            DataType::Float64 => OutputStorage::F64(vec![0.0; capacity]),
            DataType::Boolean => OutputStorage::Bool(vec![0; capacity]),
            DataType::Utf8 => OutputStorage::Str {
                ptrs: vec![0; capacity],
                lens: vec![0; capacity],
                view: false,
            },
            DataType::Utf8View => OutputStorage::Str {
                ptrs: vec![0; capacity],
                lens: vec![0; capacity],
                view: true,
            },
            other => {
                return Err(CompileError::UnsupportedType(format!(
                    "unsupported output type in raw-buffer runtime: {other:?}"
                )))
            }
        })
    }

    fn as_mut_ptr_u8(&mut self) -> *mut u8 {
        match self {
            OutputStorage::I64(v) => v.as_mut_ptr().cast::<u8>(),
            OutputStorage::U64(v) => v.as_mut_ptr().cast::<u8>(),
            OutputStorage::F64(v) => v.as_mut_ptr().cast::<u8>(),
            OutputStorage::Bool(v) => v.as_mut_ptr(),
            OutputStorage::Str { ptrs, .. } => ptrs.as_mut_ptr().cast::<u8>(),
            OutputStorage::ViewRaw { views, .. } => views.as_mut_ptr().cast::<u8>(),
        }
    }

    fn as_aux_mut_ptr_u8(&mut self) -> *mut u8 {
        match self {
            OutputStorage::Str { lens, .. } => lens.as_mut_ptr().cast::<u8>(),
            _ => ptr::null_mut(),
        }
    }

    fn truncate(&mut self, len: usize) {
        match self {
            OutputStorage::I64(v) => v.truncate(len),
            OutputStorage::U64(v) => v.truncate(len),
            OutputStorage::F64(v) => v.truncate(len),
            OutputStorage::Bool(v) => v.truncate(len),
            OutputStorage::Str { ptrs, lens, .. } => {
                ptrs.truncate(len);
                lens.truncate(len);
            }
            OutputStorage::ViewRaw { views, .. } => views.truncate(len),
        }
    }

    fn into_array(self) -> ArrayRef {
        match self {
            OutputStorage::I64(v) => Arc::new(Int64Array::from(v)),
            OutputStorage::U64(v) => Arc::new(UInt64Array::from(v)),
            OutputStorage::F64(v) => Arc::new(Float64Array::from(v)),
            OutputStorage::Bool(v) => {
                let vals = v.into_iter().map(|b| b != 0).collect::<Vec<_>>();
                Arc::new(BooleanArray::from(vals))
            }
            OutputStorage::Str { ptrs, lens, view } => {
                let vals = ptrs
                    .iter()
                    .zip(lens.iter())
                    .map(|(p, l)| {
                        let bytes = unsafe { std::slice::from_raw_parts(*p as *const u8, *l) };
                        let s = std::str::from_utf8(bytes).unwrap();
                        Some(s)
                    })
                    .collect::<Vec<_>>();
                if view {
                    Arc::new(StringViewArray::from(vals))
                } else {
                    Arc::new(StringArray::from(vals))
                }
            }
            OutputStorage::ViewRaw { views, buffers } => {
                Arc::new(StringViewArray::new(views.into(), buffers, None))
            }
        }
    }
}

pub fn execute_batch(
    kernel: super::codegen::FusedKernel,
    batch: &RecordBatch,
    program: &FusedProgram,
) -> Result<RecordBatch, CompileError> {
    let rows = batch.num_rows();

    let (input_ptrs, input_aux_ptrs, input_backing) = build_input_ptrs(batch)?;
    let literal_ptrs = program
        .string_literals
        .iter()
        .map(|s| s.as_ptr())
        .collect::<Vec<_>>();
    let literal_lens = program
        .string_literals
        .iter()
        .map(|s| s.len())
        .collect::<Vec<_>>();

    let mut output =
        allocate_output_storage(&program.output_schema, batch, &program.projection, rows)?;
    let mut output_ptrs = output
        .iter_mut()
        .map(OutputStorage::as_mut_ptr_u8)
        .collect::<Vec<_>>();
    let mut output_aux_ptrs = output
        .iter_mut()
        .map(OutputStorage::as_aux_mut_ptr_u8)
        .collect::<Vec<_>>();

    let mut write_index = 0_i64;
    let mut ctx = ExecCtx {
        input_ptrs: input_ptrs.as_ptr(),
        input_aux_ptrs: input_aux_ptrs.as_ptr(),
        output_ptrs: output_ptrs.as_mut_ptr(),
        output_aux_ptrs: output_aux_ptrs.as_mut_ptr(),
        literal_ptrs: literal_ptrs.as_ptr(),
        literal_lens: literal_lens.as_ptr(),
        write_index_ptr: (&mut write_index as *mut i64),
    };

    unsafe {
        kernel((&mut ctx as *mut ExecCtx).cast::<c_void>(), 0, rows as i64);
    }

    let out_len = write_index as usize;
    for col in &mut output {
        col.truncate(out_len);
    }

    let cols = output
        .into_iter()
        .map(OutputStorage::into_array)
        .collect::<Vec<_>>();
    drop(input_backing);
    Ok(RecordBatch::try_new(Arc::new(program.output_schema.clone()), cols).unwrap())
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum GroupValue {
    Int64(i64),
    Utf8(String),
    Bool(bool),
}

type GroupKey = Vec<GroupValue>;

fn make_group_key(group_arrays: &[ArrayRef], row_idx: usize) -> GroupKey {
    group_arrays
        .iter()
        .map(|array| {
            if array.is_null(row_idx) {
                panic!("Nulls in group keys are not supported");
            }
            match array.data_type() {
                DataType::Int64 => {
                    let array = array.as_any().downcast_ref::<Int64Array>().unwrap();
                    GroupValue::Int64(array.value(row_idx))
                }
                DataType::Utf8 => {
                    let array = array.as_any().downcast_ref::<StringArray>().unwrap();
                    GroupValue::Utf8(array.value(row_idx).to_string())
                }
                DataType::Utf8View => {
                    let array = array.as_any().downcast_ref::<StringViewArray>().unwrap();
                    GroupValue::Utf8(array.value(row_idx).to_string())
                }
                DataType::Boolean => {
                    let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                    GroupValue::Bool(array.value(row_idx))
                }
                data_type => panic!("Unsupported group key data type: {:?}", data_type),
            }
        })
        .collect()
}

fn group_columns_from_keys(keys: &[GroupKey], schema: &Schema, group_len: usize) -> Vec<ArrayRef> {
    (0..group_len)
        .map(|col_idx| {
            let field = schema.field(col_idx);
            match field.data_type() {
                DataType::Int64 => {
                    let values: Vec<Option<i64>> = keys
                        .iter()
                        .map(|key| match &key[col_idx] {
                            GroupValue::Int64(v) => Some(*v),
                            other => {
                                panic!("Group key type mismatch at column {}: {:?}", col_idx, other)
                            }
                        })
                        .collect();
                    Arc::new(Int64Array::from(values)) as ArrayRef
                }
                DataType::Utf8 => {
                    let values: Vec<Option<&str>> = keys
                        .iter()
                        .map(|key| match &key[col_idx] {
                            GroupValue::Utf8(v) => Some(v.as_str()),
                            other => {
                                panic!("Group key type mismatch at column {}: {:?}", col_idx, other)
                            }
                        })
                        .collect();
                    Arc::new(StringArray::from(values)) as ArrayRef
                }
                DataType::Utf8View => {
                    let values: Vec<Option<&str>> = keys
                        .iter()
                        .map(|key| match &key[col_idx] {
                            GroupValue::Utf8(v) => Some(v.as_str()),
                            other => {
                                panic!("Group key type mismatch at column {}: {:?}", col_idx, other)
                            }
                        })
                        .collect();
                    Arc::new(StringViewArray::from(values)) as ArrayRef
                }
                DataType::Boolean => {
                    let values: Vec<Option<bool>> = keys
                        .iter()
                        .map(|key| match &key[col_idx] {
                            GroupValue::Bool(v) => Some(*v),
                            other => {
                                panic!("Group key type mismatch at column {}: {:?}", col_idx, other)
                            }
                        })
                        .collect();
                    Arc::new(BooleanArray::from(values)) as ArrayRef
                }
                data_type => panic!("Unsupported group output data type: {:?}", data_type),
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
enum AggKind {
    Count,
    Sum,
    Min,
    Max,
    Avg,
}

#[derive(Clone)]
enum PartialAggState {
    Count(u64),
    Sum(i128),
    Min(Option<i64>),
    Max(Option<i64>),
    Avg { sum: i128, count: u64 },
}

impl PartialAggState {
    fn new(kind: AggKind) -> Self {
        match kind {
            AggKind::Count => PartialAggState::Count(0),
            AggKind::Sum => PartialAggState::Sum(0),
            AggKind::Min => PartialAggState::Min(None),
            AggKind::Max => PartialAggState::Max(None),
            AggKind::Avg => PartialAggState::Avg { sum: 0, count: 0 },
        }
    }

    fn merge_from(&mut self, other: PartialAggState) {
        match (self, other) {
            (PartialAggState::Count(a), PartialAggState::Count(b)) => *a += b,
            (PartialAggState::Sum(a), PartialAggState::Sum(b)) => *a += b,
            (PartialAggState::Min(a), PartialAggState::Min(b)) => match (*a, b) {
                (Some(x), Some(y)) => *a = Some(x.min(y)),
                (None, Some(y)) => *a = Some(y),
                _ => {}
            },
            (PartialAggState::Max(a), PartialAggState::Max(b)) => match (*a, b) {
                (Some(x), Some(y)) => *a = Some(x.max(y)),
                (None, Some(y)) => *a = Some(y),
                _ => {}
            },
            (
                PartialAggState::Avg {
                    sum: a_sum,
                    count: a_cnt,
                },
                PartialAggState::Avg {
                    sum: b_sum,
                    count: b_cnt,
                },
            ) => {
                *a_sum += b_sum;
                *a_cnt += b_cnt;
            }
            _ => panic!("partial aggregate state mismatch while merging"),
        }
    }
}

struct PartialAggChunk {
    key_order: Vec<GroupKey>,
    map: HashMap<GroupKey, Vec<PartialAggState>>,
}

fn agg_kind(expr: &AggregateExpr) -> AggKind {
    match expr {
        AggregateExpr::Count(_) => AggKind::Count,
        AggregateExpr::Sum(_) => AggKind::Sum,
        AggregateExpr::Min(_) => AggKind::Min,
        AggregateExpr::Max(_) => AggKind::Max,
        AggregateExpr::Average(_) => AggKind::Avg,
    }
}

fn process_agg_chunk(
    group_arrays: &[ArrayRef],
    agg_input_arrays: &[ArrayRef],
    kinds: &[AggKind],
    start: usize,
    end: usize,
) -> PartialAggChunk {
    let int_inputs = agg_input_arrays
        .iter()
        .map(|a| a.as_any().downcast_ref::<Int64Array>())
        .collect::<Vec<_>>();
    let mut map: HashMap<GroupKey, Vec<PartialAggState>> = HashMap::new();
    let mut key_order = Vec::new();

    for row_idx in start..end {
        let key = make_group_key(group_arrays, row_idx);
        let states = map.entry(key.clone()).or_insert_with(|| {
            key_order.push(key);
            kinds.iter().map(|k| PartialAggState::new(*k)).collect()
        });

        for (agg_idx, state) in states.iter_mut().enumerate() {
            let input = &agg_input_arrays[agg_idx];
            match state {
                PartialAggState::Count(v) => {
                    if !input.is_null(row_idx) {
                        *v += 1;
                    }
                }
                PartialAggState::Sum(v) => {
                    let arr = int_inputs[agg_idx].unwrap_or_else(|| {
                        panic!("SUM expects Int64 input, got {:?}", input.data_type())
                    });
                    if !arr.is_null(row_idx) {
                        *v += arr.value(row_idx) as i128;
                    }
                }
                PartialAggState::Min(v) => {
                    let arr = int_inputs[agg_idx].unwrap_or_else(|| {
                        panic!("MIN expects Int64 input, got {:?}", input.data_type())
                    });
                    if !arr.is_null(row_idx) {
                        let cur = arr.value(row_idx);
                        *v = Some(v.map_or(cur, |x| x.min(cur)));
                    }
                }
                PartialAggState::Max(v) => {
                    let arr = int_inputs[agg_idx].unwrap_or_else(|| {
                        panic!("MAX expects Int64 input, got {:?}", input.data_type())
                    });
                    if !arr.is_null(row_idx) {
                        let cur = arr.value(row_idx);
                        *v = Some(v.map_or(cur, |x| x.max(cur)));
                    }
                }
                PartialAggState::Avg { sum, count } => {
                    let arr = int_inputs[agg_idx].unwrap_or_else(|| {
                        panic!("AVG expects Int64 input, got {:?}", input.data_type())
                    });
                    if !arr.is_null(row_idx) {
                        *sum += arr.value(row_idx) as i128;
                        *count += 1;
                    }
                }
            }
        }
    }

    PartialAggChunk { key_order, map }
}

fn merge_partial_chunk(
    global_map: &mut HashMap<GroupKey, Vec<PartialAggState>>,
    global_key_order: &mut Vec<GroupKey>,
    chunk: PartialAggChunk,
) {
    let mut local_map = chunk.map;
    for key in chunk.key_order {
        let local_states = local_map.remove(&key).unwrap();
        if let Some(global_states) = global_map.get_mut(&key) {
            for (dst, src) in global_states.iter_mut().zip(local_states.into_iter()) {
                dst.merge_from(src);
            }
        } else {
            global_key_order.push(key.clone());
            global_map.insert(key, local_states);
        }
    }
}

fn aggregate_columns_from_states(
    key_order: &[GroupKey],
    map: &HashMap<GroupKey, Vec<PartialAggState>>,
    kinds: &[AggKind],
) -> Vec<ArrayRef> {
    kinds
        .iter()
        .enumerate()
        .map(|(agg_idx, kind)| match kind {
            AggKind::Count => {
                let values = key_order
                    .iter()
                    .map(|k| match &map.get(k).unwrap()[agg_idx] {
                        PartialAggState::Count(v) => Some(*v),
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                Arc::new(UInt64Array::from(values)) as ArrayRef
            }
            AggKind::Sum => {
                let values = key_order
                    .iter()
                    .map(|k| match &map.get(k).unwrap()[agg_idx] {
                        PartialAggState::Sum(v) => Some(*v as i64),
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                Arc::new(Int64Array::from(values)) as ArrayRef
            }
            AggKind::Min => {
                let values = key_order
                    .iter()
                    .map(|k| match &map.get(k).unwrap()[agg_idx] {
                        PartialAggState::Min(v) => *v,
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                Arc::new(Int64Array::from(values)) as ArrayRef
            }
            AggKind::Max => {
                let values = key_order
                    .iter()
                    .map(|k| match &map.get(k).unwrap()[agg_idx] {
                        PartialAggState::Max(v) => *v,
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                Arc::new(Int64Array::from(values)) as ArrayRef
            }
            AggKind::Avg => {
                let values = key_order
                    .iter()
                    .map(|k| match &map.get(k).unwrap()[agg_idx] {
                        PartialAggState::Avg { sum, count } => {
                            if *count == 0 {
                                None
                            } else {
                                Some(*sum as f64 / *count as f64)
                            }
                        }
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                Arc::new(Float64Array::from(values)) as ArrayRef
            }
        })
        .collect()
}

pub fn execute_hash_aggregate(
    batches: &[RecordBatch],
    group_expr: &[PhysicalExpr],
    aggregate_expr: &[AggregateExpr],
    output_schema: &Schema,
) -> Vec<RecordBatch> {
    let kinds = aggregate_expr.iter().map(agg_kind).collect::<Vec<_>>();
    let mut map: HashMap<GroupKey, Vec<PartialAggState>> = HashMap::new();
    let mut key_order = Vec::new();

    for batch in batches {
        let group_arrays: Vec<_> = group_expr.iter().map(|e| e.evaluate(batch)).collect();
        let agg_input_arrays: Vec<_> = aggregate_expr
            .iter()
            .map(|e| e.expression().evaluate(batch))
            .collect();

        let partial = process_agg_chunk(
            &group_arrays,
            &agg_input_arrays,
            &kinds,
            0,
            batch.num_rows(),
        );
        merge_partial_chunk(&mut map, &mut key_order, partial);
    }

    if key_order.is_empty() {
        if aggregate_expr.is_empty() {
            return vec![];
        }
        if group_expr.is_empty() {
            let key = Vec::new();
            map.insert(
                key.clone(),
                kinds.iter().map(|k| PartialAggState::new(*k)).collect(),
            );
            key_order.push(key);
        } else {
            return vec![];
        }
    }

    let group_len = group_expr.len();
    let mut columns = group_columns_from_keys(&key_order, output_schema, group_len);
    columns.extend(aggregate_columns_from_states(&key_order, &map, &kinds));

    let batch = RecordBatch::try_new(Arc::new(output_schema.clone()), columns).unwrap();
    vec![batch]
}

fn allocate_output_storage(
    schema: &Schema,
    batch: &RecordBatch,
    projection: &[super::lowering::FusedExpr],
    rows: usize,
) -> Result<Vec<OutputStorage>, CompileError> {
    let mut out = Vec::with_capacity(schema.fields().len());
    for (i, f) in schema.fields().iter().enumerate() {
        let dtype = f.data_type();
        if matches!(dtype, DataType::Utf8View) {
            if let Some(col_idx) = projection.get(i).and_then(|expr| expr.as_column_index()) {
                if let Some(arr) = batch
                    .column(col_idx)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                {
                    out.push(OutputStorage::ViewRaw {
                        views: vec![0; rows],
                        buffers: arr.data_buffers().to_vec(),
                    });
                    continue;
                }
            }
        }
        out.push(OutputStorage::new(dtype, rows)?);
    }
    Ok(out)
}

type InputPtrTables = (Vec<*const u8>, Vec<*const u8>, Vec<InputBacking>);

fn build_input_ptrs(batch: &RecordBatch) -> Result<InputPtrTables, CompileError> {
    let mut ptrs = Vec::with_capacity(batch.num_columns());
    let mut aux_ptrs = Vec::with_capacity(batch.num_columns());
    let mut backing = Vec::new();

    for col in batch.columns() {
        match col.data_type() {
            DataType::Int64 => {
                let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                ptrs.push(arr.values().as_ptr().cast::<u8>());
                aux_ptrs.push(ptr::null());
            }
            DataType::UInt64 => {
                let arr = col.as_any().downcast_ref::<UInt64Array>().unwrap();
                ptrs.push(arr.values().as_ptr().cast::<u8>());
                aux_ptrs.push(ptr::null());
            }
            DataType::Float64 => {
                let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                ptrs.push(arr.values().as_ptr().cast::<u8>());
                aux_ptrs.push(ptr::null());
            }
            DataType::Boolean => {
                let arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
                let bytes = (0..arr.len())
                    .map(|i| if arr.value(i) { 1u8 } else { 0u8 })
                    .collect::<Vec<_>>();
                ptrs.push(bytes.as_ptr());
                aux_ptrs.push(ptr::null());
                backing.push(InputBacking::BoolBytes { _bytes: bytes });
            }
            DataType::Utf8 => {
                let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
                ptrs.push(arr.value_offsets().as_ptr().cast::<u8>());
                aux_ptrs.push(arr.value_data().as_ptr());
            }
            DataType::Utf8View => {
                let arr = col.as_any().downcast_ref::<StringViewArray>().unwrap();
                let buffer_ptrs = arr
                    .data_buffers()
                    .iter()
                    .map(|b| b.as_ptr() as usize)
                    .collect::<Vec<_>>();
                ptrs.push(arr.views().as_ptr().cast::<u8>());
                aux_ptrs.push(buffer_ptrs.as_ptr().cast::<u8>());
                backing.push(InputBacking::ViewBufferPtrs { _ptrs: buffer_ptrs });
            }
            other => {
                return Err(CompileError::UnsupportedType(format!(
                    "unsupported input type in raw-buffer runtime: {other:?}"
                )));
            }
        }
    }

    Ok((ptrs, aux_ptrs, backing))
}

#[no_mangle]
pub extern "C" fn jit_str_cmp(
    left_ptr: *const u8,
    left_len: i64,
    right_ptr: *const u8,
    right_len: i64,
    op: i64,
) -> i8 {
    let left = unsafe { std::slice::from_raw_parts(left_ptr, left_len as usize) };
    let right = unsafe { std::slice::from_raw_parts(right_ptr, right_len as usize) };
    let ord = left.cmp(right);
    let matched = match op {
        0 => ord.is_gt(),
        1 => ord.is_ge(),
        2 => ord.is_lt(),
        3 => ord.is_le(),
        4 => ord.is_eq(),
        5 => ord.is_ne(),
        _ => false,
    };

    i8::from(matched)
}

#[no_mangle]
pub extern "C" fn jit_memcmp_eq(left_ptr: *const u8, right_ptr: *const u8, len: i64) -> i8 {
    let left = unsafe { std::slice::from_raw_parts(left_ptr, len as usize) };
    let right = unsafe { std::slice::from_raw_parts(right_ptr, len as usize) };
    i8::from(left == right)
}

use crate::codegen::backend::SemValue;

#[derive(Clone, Copy, Debug)]
pub(super) enum SemGenValue {
    I64(SemValue),
    U64(SemValue),
    F64(SemValue),
    BoolI8(SemValue),
    Str { ptr: SemValue, len: SemValue },
}

impl SemGenValue {
    pub(super) fn to_bool_i8(self) -> SemValue {
        match self {
            SemGenValue::BoolI8(v) => v,
            _ => panic!("expected Boolean value"),
        }
    }
}

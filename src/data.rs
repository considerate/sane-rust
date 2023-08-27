use ndarray::ArrayD;
use quickcheck::{Arbitrary, Gen};

/// SANE [supported data types](https://github.com/considerate/sane#data-types)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    I32,
    U32,
    F64,
    I64,
    U64,
    I8,
    U8,
}

impl Arbitrary for DataType {
    fn arbitrary(gen: &mut Gen) -> Self {
        use DataType::*;
        let options = [F32, I32, U32, F64, I64, U64, I8, U8];
        gen.choose(&options).unwrap().clone()
    }
}

/// Parse a SANE-encoded u8 into the corresponding [`DataType`].
pub fn parse_data_type(code: u8) -> Result<DataType, u8> {
    match code {
        0 => Ok(DataType::F32),
        1 => Ok(DataType::I32),
        2 => Ok(DataType::U32),
        3 => Ok(DataType::F64),
        4 => Ok(DataType::I64),
        5 => Ok(DataType::U64),
        6 => Ok(DataType::I8),
        7 => Ok(DataType::U8),
        n => Err(n),
    }
}

/// Get the `u8` SANE-encoding of a [`DataType`].
pub fn data_type_code(data_type: DataType) -> u8 {
    match data_type {
        DataType::F32 => 0,
        DataType::I32 => 1,
        DataType::U32 => 2,
        DataType::F64 => 3,
        DataType::I64 => 4,
        DataType::U64 => 5,
        DataType::I8 => 6,
        DataType::U8 => 7,
    }
}

/// A Sane array is an array with dynamic shape and elements of one of the [supported data
/// types](https://github.com/considerate/sane#data-types)
pub enum Sane {
    ArrayF32(ArrayD<f32>),
    ArrayI32(ArrayD<i32>),
    ArrayU32(ArrayD<u32>),
    ArrayF64(ArrayD<f64>),
    ArrayI64(ArrayD<i64>),
    ArrayU64(ArrayD<u64>),
    ArrayI8(ArrayD<i8>),
    ArrayU8(ArrayD<u8>),
}


/// The header of a SANE array, consisting of the shape, the data type and the length of the data
/// in number of bytes
pub struct Header {
    pub shape: Vec<usize>,
    pub data_type: DataType,
    pub data_length: usize,
}

pub trait SaneData: Copy {
    fn sane_data_type() -> DataType;
}

impl SaneData for f32 {
    fn sane_data_type()  -> DataType {
        DataType::F32
    }
}

impl SaneData for i32 {
    fn sane_data_type()  -> DataType {
        DataType::I32
    }
}

impl SaneData for u32 {
    fn sane_data_type()  -> DataType {
        DataType::U32
    }
}

impl SaneData for f64 {
    fn sane_data_type()  -> DataType {
        DataType::F64
    }
}

impl SaneData for i64 {
    fn sane_data_type()  -> DataType {
        DataType::I64
    }
}

impl SaneData for u64 {
    fn sane_data_type()  -> DataType {
        DataType::U64
    }
}

impl SaneData for i8 {
    fn sane_data_type()  -> DataType {
        DataType::I8
    }
}

impl SaneData for u8 {
    fn sane_data_type()  -> DataType {
        DataType::U8
    }
}

#[cfg(test)]
mod tests {
    use super::{DataType, parse_data_type, data_type_code};
    use quickcheck::quickcheck;
    quickcheck! {
        fn prop_data_type_round(data_type: DataType) -> bool {
            Ok(data_type.clone()) == parse_data_type(data_type_code(data_type))
        }
    }
}

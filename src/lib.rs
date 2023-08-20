use std::io::prelude::{Read, Write};
use std::mem::size_of;
use std::num::TryFromIntError;
use std::slice::from_raw_parts;
use std::error::Error;

use ndarray::{IxDyn, ArrayView, ArrayD, Array, Dimension, ShapeError};
use quickcheck::{Arbitrary, Gen};

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

pub fn get_data_type(code: u8) -> Result<DataType, u8> {
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


struct Header {
    shape: Vec<usize>,
    data_type: DataType,
    data_length: usize,
}

#[derive(Debug)]
pub enum ParseError {
    NotSANE,
    InvalidDataType(u8),
    NotEnoughBytes(std::io::Error),
    CannotConvertToUSize(TryFromIntError),
    ReadError(std::io::Error),
    ShapeError(ShapeError),
    WrongDataType(DataType),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ParseError::*;
        match self {
            NotSANE => write!(f, "Not a SANE array"),
            InvalidDataType(code) => write!(f, "Invalid data type code: {}", code),
            NotEnoughBytes(err) => write!(f, "Not enough bytes: {}", err),
            CannotConvertToUSize(err) => write!(f, "Cannot convert to size: {}", err),
            ReadError(err) => write!(f, "Failed to read: {}", err),
            ShapeError(err) => write!(f, "{}", err),
            WrongDataType(t) => write!(f, "unexpected data type {:?}", t),
        }
    }
}

fn parse_u32_size(bytes: [u8; 4]) -> Result<usize, ParseError> {
    usize::try_from(u32::from_le_bytes(bytes)).map_err(ParseError::CannotConvertToUSize)
}

fn parse_u64_size(bytes: [u8; 8]) -> Result<usize, ParseError> {
    usize::try_from(u64::from_le_bytes(bytes)).map_err(ParseError::CannotConvertToUSize)
}

fn read_header<F: Read>(mut file: F) -> Result<(Header, F), ParseError> {
    let mut magic_bytes = [0; 4];
    file.read_exact(&mut magic_bytes).map_err(ParseError::NotEnoughBytes)?;
    let sane_bytes = "SANE".as_bytes();
    if magic_bytes != sane_bytes {
        return Err(ParseError::NotSANE);
    }
    let mut shape_length_bytes = [0; 4];
    file.read_exact(&mut shape_length_bytes).map_err(ParseError::NotEnoughBytes)?;
    let shape_length = parse_u32_size(shape_length_bytes)?;
    let mut shape_bytes = vec![0u8; shape_length * 8];
    file.read_exact(&mut shape_bytes).map_err(ParseError::NotEnoughBytes)?;
    let mut shape = vec![];
    for dim in 0..shape_length {
        let mut dim_bytes = [0; 8];
        dim_bytes.copy_from_slice(&shape_bytes[dim * 8..(dim+1)*8]);
        let dimension = parse_u64_size(dim_bytes)?;
        shape.push(dimension);
    }
    shape.reverse();
    let mut data_type_bytes = [0; 1];
    file.read_exact(&mut data_type_bytes).map_err(ParseError::NotEnoughBytes)?;
    let data_type = get_data_type(data_type_bytes[0]).map_err(ParseError::InvalidDataType)?;
    let mut data_length_bytes = [0; 8];
    file.read_exact(&mut data_length_bytes).map_err(ParseError::NotEnoughBytes)?;
    let data_length = parse_u64_size(data_length_bytes)?;
    Ok((Header {
        shape,
        data_type,
        data_length,
    }, file))
}

fn align_array<T: SaneData>(dims: IxDyn, byte_data: Vec<u8>) -> Result<ArrayD<T>, ParseError> {
    if cfg!(endianness = "little") {
        let values = unsafe {
            byte_data.align_to::<T>().1
        };
        let array_view = ArrayView::from_shape(dims, &values).map_err(ParseError::ShapeError)?;
        Ok(array_view.to_owned())
    } else {
        let vec = T::from_le_bytes(byte_data);
        let array_view = ArrayView::from_shape(dims, &vec).map_err(ParseError::ShapeError)?;
        Ok(array_view.to_owned())
    }
}

fn align_array_shape<T: SaneData, D: Dimension>(shape: Vec<usize>, byte_data: Vec<u8>) -> Result<Array<T,D>, ParseError> {
    let dyn_dims = IxDyn(&shape);
    if cfg!(endianness = "little") {
        let values = unsafe {
            byte_data.align_to::<T>().1
        };
        let array_view = ArrayView::from_shape(dyn_dims, &values).map_err(ParseError::ShapeError)?;
        let shaped_array = array_view.into_dimensionality().map_err(ParseError::ShapeError)?;
        Ok(shaped_array.to_owned())
    } else {
        let values = T::from_le_bytes(byte_data);
        let array_view = ArrayView::from_shape(dyn_dims, &values).map_err(ParseError::ShapeError)?;
        let shaped_array = array_view.into_dimensionality().map_err(ParseError::ShapeError)?;
        Ok(shaped_array.to_owned())
    }
}

pub fn read_sane<F: Read, A: SaneData, D: Dimension>(file: F) -> Result<(Array<A, D>, F), ParseError> {
    let (header, mut file) = read_header(file)?;
    let mut sane_data = vec![0u8; header.data_length];
    file.read_exact(&mut sane_data).map_err(ParseError::NotEnoughBytes)?;
    if header.data_type != A::sane_data_type() {
        Err(ParseError::WrongDataType(header.data_type))?;
    }
    let sane = align_array_shape(header.shape, sane_data)?;
    Ok((sane, file))
}


pub fn read_sane_dyn<F: Read>(file: F) -> Result<(Sane, F), ParseError> {
    let (header, mut file) = read_header(file)?;
    let mut sane_data = vec![0u8; header.data_length];
    file.read_exact(&mut sane_data).map_err(ParseError::NotEnoughBytes)?;
    let dims: IxDyn = IxDyn(&header.shape);
    let sane = match header.data_type {
        DataType::F32 => align_array(dims, sane_data).map(Sane::ArrayF32),
        DataType::I32 => align_array(dims, sane_data).map(Sane::ArrayI32),
        DataType::U32 => align_array(dims, sane_data).map(Sane::ArrayU32),
        DataType::F64 => align_array(dims, sane_data).map(Sane::ArrayF64),
        DataType::I64 => align_array(dims, sane_data).map(Sane::ArrayI64),
        DataType::U64 => align_array(dims, sane_data).map(Sane::ArrayU64),
        DataType::I8 => align_array(dims, sane_data).map(Sane::ArrayI8),
        DataType::U8 => align_array(dims, sane_data).map(Sane::ArrayU8),
    }?;
    Ok((sane, file))
}


pub trait SaneData: Copy {
    fn sane_data_type() -> DataType;
    fn to_le_bytes(elem: Self) -> Vec<u8>;
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<Self>;
}

macro_rules! sane_from_le_bytes {
    ($t:ty, $e:expr) => {
        {
            const COUNT: usize = std::mem::size_of::<$t>();
            let elems = $e.len() / COUNT;
            let mut result = vec![];
            for i in 0..elems {
                let elem_bytes: [u8; COUNT] = $e[i*COUNT..(i+1)*COUNT].try_into().unwrap();
                result.push(<$t>::from_le_bytes(elem_bytes));
            };
            result
        }
    }
}

impl SaneData for f32 {
    fn sane_data_type()  -> DataType {
        DataType::F32
    }
    fn to_le_bytes(elem: f32) -> Vec<u8> {
        f32::to_le_bytes(elem).to_vec()
    }
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<f32> {
        return sane_from_le_bytes!(f32, bytes);
    }
}

impl SaneData for i32 {
    fn sane_data_type()  -> DataType {
        DataType::I32
    }
    fn to_le_bytes(elem: i32) -> Vec<u8> {
        i32::to_le_bytes(elem).to_vec()
    }
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<i32> {
        return sane_from_le_bytes!(i32, bytes);
    }
}

impl SaneData for u32 {
    fn sane_data_type()  -> DataType {
        DataType::U32
    }
    fn to_le_bytes(elem: u32) -> Vec<u8> {
        u32::to_le_bytes(elem).to_vec()
    }
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<u32> {
        return sane_from_le_bytes!(u32, bytes);
    }
}

impl SaneData for f64 {
    fn sane_data_type()  -> DataType {
        DataType::F64
    }
    fn to_le_bytes(elem: f64) -> Vec<u8> {
        f64::to_le_bytes(elem).to_vec()
    }
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<f64> {
        return sane_from_le_bytes!(f64, bytes);
    }
}

impl SaneData for i64 {
    fn sane_data_type()  -> DataType {
        DataType::I64
    }
    fn to_le_bytes(elem: i64) -> Vec<u8> {
        i64::to_le_bytes(elem).to_vec()
    }
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<i64> {
        return sane_from_le_bytes!(i64, bytes);
    }
}

impl SaneData for u64 {
    fn sane_data_type()  -> DataType {
        DataType::U64
    }
    fn to_le_bytes(elem: u64) -> Vec<u8> {
        u64::to_le_bytes(elem).to_vec()
    }
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<u64> {
        return sane_from_le_bytes!(u64, bytes);
    }
}

impl SaneData for i8 {
    fn sane_data_type()  -> DataType {
        DataType::I8
    }
    fn to_le_bytes(elem: i8) -> Vec<u8> {
        i8::to_le_bytes(elem).to_vec()
    }
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<i8> {
        return sane_from_le_bytes!(i8, bytes);
    }
}

impl SaneData for u8 {
    fn sane_data_type()  -> DataType {
        DataType::U8
    }
    fn to_le_bytes(elem: u8) -> Vec<u8> {
        vec![elem]
    }
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<u8> {
        return bytes;
    }
}

#[derive(Debug)]
pub enum WriteError {
    Failed(std::io::Error),
    ShapeTooLong(<u32 as TryFrom<usize>>::Error),
    DimTooLarge(<u64 as TryFrom<usize>>::Error),
    TooMuchData(<u64 as TryFrom<usize>>::Error),
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use WriteError::*;
        match self {
            Failed(e) => write!(f, "Failed to write {}", e),
            ShapeTooLong(e) => write!(f, "Shape length doesn't fit in 32 bits {}", e),
            DimTooLarge(e) => write!(f, "Dimension size doesn't fit in 64 bits {}", e),
            TooMuchData(e) => write!(f, "Length of array doesn't fit in 64 bits {}", e),
        }
    }
}

impl Error for WriteError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

fn write_header<F: Write, A: SaneData, D: Dimension>(file: &mut F, array: &Array<A, D>)  -> Result<(), WriteError> {
    let shape = array.shape();
    let data_type = A::sane_data_type();
    let magic = "SANE".as_bytes();
    file.write_all(magic).map_err(WriteError::Failed)?;
    let shape_length = u32::try_from(shape.len()).map_err(WriteError::ShapeTooLong)?;
    let shape_length_bytes = shape_length.to_le_bytes();
    file.write_all(&shape_length_bytes).map_err(WriteError::Failed)?;
    for &dim in shape.iter().rev() {
        let dimension = u64::try_from(dim).map_err(WriteError::DimTooLarge)?;
        let dim_bytes = dimension.to_le_bytes();
        file.write_all(&dim_bytes).map_err(WriteError::Failed)?
    }
    let code = data_type_code(data_type);
    file.write_all(&[code]).map_err(WriteError::Failed)?;
    let byte_length = array.len() * size_of::<A>();
    let data_length = u64::try_from(byte_length).map_err(WriteError::TooMuchData)?;
    let data_length_bytes = data_length.to_le_bytes();
    file.write_all(&data_length_bytes).map_err(WriteError::Failed)?;
    Ok(())
}

fn write_data<F: Write, A: SaneData, D: Dimension>(file: &mut F, array: &Array<A, D>) -> Result<(), WriteError> {
    let data_ptr = array.as_ptr();
    let byte_length = array.len() * size_of::<A>();
    if cfg!(endianness = "little") {
        let data_ptr_bytes = data_ptr.cast::<u8>();
        let data_bytes = unsafe { from_raw_parts(data_ptr_bytes, byte_length) };
        file.write_all(data_bytes).map_err(WriteError::Failed)?;
    } else {
        for &elem in array.iter() {
            let elem_bytes = SaneData::to_le_bytes(elem);
            file.write_all(&elem_bytes).map_err(WriteError::Failed)?;
        }
    }
    Ok(())
}

pub fn write_sane<F: Write, A: SaneData,D: Dimension>(mut file: F, array: Array<A, D>) -> Result<(), WriteError> {
    write_header(&mut file, &array)?;
    write_data(&mut file, &array)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    extern crate quickcheck;
    use quickcheck::quickcheck;
    use std::io::Cursor;

    use super::*;

    quickcheck! {
        fn prop_data_type_round(data_type: DataType) -> bool {
            Ok(data_type.clone()) == get_data_type(data_type_code(data_type))
        }
    }

    #[test]
    fn example_roundtrip() {
        let arr = ndarray::array![[1,2,3], [4,5,6]];
        let mut file = Cursor::new(Vec::new());
        write_sane(&mut file, arr.clone()).unwrap();
        file.set_position(0);
        let (parsed, _) = read_sane_dyn(file).unwrap();
        match parsed {
            Sane::ArrayI32(arr2) => assert_eq!(arr.into_dyn(), arr2),
            _ => assert!(false),
        }
    }

    #[test]
    fn example_roundtrip_typed() {
        let arr = ndarray::array![[1,2,3], [4,5,6]];
        let mut file = Cursor::new(Vec::new());
        write_sane(&mut file, arr.clone()).unwrap();
        file.set_position(0);
        let (arr2, _) = read_sane(file).unwrap();
        assert_eq!(arr, arr2)
    }

    #[test]
    fn example_roundtrip_typed_wrong_shape() {
        let arr = ndarray::array![[1,2,3], [4,5,6]];
        let wrong = ndarray::array![[[1,2,3], [4,5,6]]];
        let mut file = Cursor::new(Vec::new());
        write_sane(&mut file, arr.clone()).unwrap();
        file.set_position(0);
        match read_sane(file) {
            Ok((actual, _)) => assert_ne!(wrong, actual), // This is here to determine the expected
                                                          // type of read_sane
            Err(ParseError::ShapeError(_)) => assert!(true),
            Err(_) => assert!(false),
        }
    }
}

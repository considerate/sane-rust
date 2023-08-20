use std::io::prelude::Read;
use std::num::TryFromIntError;

use ndarray::{IxDyn, ArrayView, ArrayD, Array, Dimension, ShapeError};
use crate::data::{DataType, SaneData, Sane, Header, get_data_type};

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

pub trait ReadSane: SaneData {
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<Self>;
}

impl ReadSane for f32 {
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<f32> {
        return sane_from_le_bytes!(f32, bytes);
    }
}

impl ReadSane for i32 {
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<i32> {
        return sane_from_le_bytes!(i32, bytes);
    }
}

impl ReadSane for u32 {
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<u32> {
        return sane_from_le_bytes!(u32, bytes);
    }
}

impl ReadSane for f64 {
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<f64> {
        return sane_from_le_bytes!(f64, bytes);
    }
}

impl ReadSane for i64 {
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<i64> {
        return sane_from_le_bytes!(i64, bytes);
    }
}

impl ReadSane for u64 {
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<u64> {
        return sane_from_le_bytes!(u64, bytes);
    }
}

impl ReadSane for i8 {
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<i8> {
        return sane_from_le_bytes!(i8, bytes);
    }
}

impl ReadSane for u8 {
    fn from_le_bytes(bytes: Vec<u8>) -> Vec<u8> {
        return bytes;
    }
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

fn read_header<F: Read>(file: &mut F) -> Result<Header, ParseError> {
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
    Ok(Header {
        shape,
        data_type,
        data_length,
    })
}

fn align_array<T: ReadSane>(dims: IxDyn, byte_data: Vec<u8>) -> Result<ArrayD<T>, ParseError> {
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

fn align_array_shape<T: ReadSane, D: Dimension>(shape: Vec<usize>, byte_data: Vec<u8>) -> Result<Array<T,D>, ParseError> {
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

pub fn read_sane<F: Read, A: ReadSane, D: Dimension>(file: &mut F) -> Result<Array<A, D>, ParseError> {
    let header = read_header(file)?;
    let mut sane_data = vec![0u8; header.data_length];
    file.read_exact(&mut sane_data).map_err(ParseError::NotEnoughBytes)?;
    if header.data_type != A::sane_data_type() {
        Err(ParseError::WrongDataType(header.data_type))?;
    }
    let sane = align_array_shape(header.shape, sane_data)?;
    Ok(sane)
}


pub fn read_sane_dyn<F: Read>(file: &mut F) -> Result<Sane, ParseError> {
    let header = read_header(file)?;
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
    Ok(sane)
}

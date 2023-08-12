use std::fs::File;
use std::io::prelude::Read;
use std::num::TryFromIntError;

use ndarray::{IxDyn, ArrayView, ArrayD};

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

enum DataType {
    F32,
    I32,
    U32,
    F64,
    I64,
    U64,
    I8,
    U8,
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

pub enum ParseError {
    NotSANE,
    InvalidDataType(u8),
    NotEnoughBytes(std::io::Error),
    CannotConvertToUSize(TryFromIntError),
    ReadError(std::io::Error),
    ShapeError(ndarray::ShapeError),
    NotEnoughData(usize, usize),
}

fn parse_u32_size(bytes: [u8; 4]) -> Result<usize, ParseError> {
    usize::try_from(u32::from_le_bytes(bytes)).map_err(ParseError::CannotConvertToUSize)
}

fn parse_u64_size(bytes: [u8; 8]) -> Result<usize, ParseError> {
    usize::try_from(u64::from_le_bytes(bytes)).map_err(ParseError::CannotConvertToUSize)
}

fn read_header(mut file: File) -> Result<(Header, File), ParseError> {
    let mut magic_bytes = [0; 4];
    file.read_exact(&mut magic_bytes).map_err(ParseError::NotEnoughBytes)?;
    if magic_bytes != [0x53, 0x41, 0x4E, 0x45] {
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
    let mut data_type_bytes = [0; 1];
    file.read_exact(&mut data_type_bytes).map_err(ParseError::NotEnoughBytes)?;
    let data_type = match data_type_bytes[0]{
        0 => Ok(DataType::F32),
        1 => Ok(DataType::I32),
        2 => Ok(DataType::U32),
        3 => Ok(DataType::F64),
        4 => Ok(DataType::I64),
        5 => Ok(DataType::U64),
        6 => Ok(DataType::I8),
        7 => Ok(DataType::U8),
        n => Err(ParseError::InvalidDataType(n)),
    }?;
    let mut data_length_bytes = [0; 8];
    file.read_exact(&mut data_length_bytes).map_err(ParseError::NotEnoughBytes)?;
    let data_length = parse_u64_size(data_length_bytes)?;
    Ok((Header {
        shape,
        data_type,
        data_length,
    }, file))
}

fn align_array<T: Clone>(dims: IxDyn, byte_data: Vec<u8>) -> Result<ArrayD<T>, ParseError> {
    let values = unsafe {
        byte_data.align_to::<T>().1
    };
    let array_view = ArrayView::from_shape(dims, &values).map_err(ParseError::ShapeError)?;
    Ok(array_view.to_owned())
}


pub fn read_sane(file: File) -> Result<Sane, ParseError> {
    let (header, mut file) = read_header(file)?;
    let mut sane_data = vec![];
    let bytes_read = file.read_to_end(&mut sane_data).map_err(ParseError::ReadError)?;
    if bytes_read < header.data_length {
        Err(ParseError::NotEnoughData(header.data_length, bytes_read))?;
    }
    // TODO: check bytes_read is (at least) the expected number of bytes?
    let dims: IxDyn = IxDyn(&header.shape);
    match header.data_type {
        DataType::F32 => align_array(dims, sane_data).map(Sane::ArrayF32),
        DataType::I32 => align_array(dims, sane_data).map(Sane::ArrayI32),
        DataType::U32 => align_array(dims, sane_data).map(Sane::ArrayU32),
        DataType::F64 => align_array(dims, sane_data).map(Sane::ArrayF64),
        DataType::I64 => align_array(dims, sane_data).map(Sane::ArrayI64),
        DataType::U64 => align_array(dims, sane_data).map(Sane::ArrayU64),
        DataType::I8 => align_array(dims, sane_data).map(Sane::ArrayI8),
        DataType::U8 => align_array(dims, sane_data).map(Sane::ArrayU8),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

}

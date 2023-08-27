use std::io::prelude::Write;
use std::mem::size_of;
use std::slice::from_raw_parts;
use std::error::Error;

use ndarray::{Dimension, ArrayBase, Data};

use crate::data::{SaneData, data_type_code};

/// To be able to write SANE data we need to be able to
/// convert an element to a byte sequence
pub trait WriteSane: SaneData {
    fn to_le_bytes(elem: Self) -> Vec<u8>;
}

impl WriteSane for f32 {
    fn to_le_bytes(elem: f32) -> Vec<u8> {
        f32::to_le_bytes(elem).to_vec()
    }
}

impl WriteSane for i32 {
    fn to_le_bytes(elem: i32) -> Vec<u8> {
        i32::to_le_bytes(elem).to_vec()
    }
}

impl WriteSane for u32 {
    fn to_le_bytes(elem: u32) -> Vec<u8> {
        u32::to_le_bytes(elem).to_vec()
    }
}

impl WriteSane for f64 {
    fn to_le_bytes(elem: f64) -> Vec<u8> {
        f64::to_le_bytes(elem).to_vec()
    }
}

impl WriteSane for i64 {
    fn to_le_bytes(elem: i64) -> Vec<u8> {
        i64::to_le_bytes(elem).to_vec()
    }
}

impl WriteSane for u64 {
    fn to_le_bytes(elem: u64) -> Vec<u8> {
        u64::to_le_bytes(elem).to_vec()
    }
}

impl WriteSane for i8 {
    fn to_le_bytes(elem: i8) -> Vec<u8> {
        i8::to_le_bytes(elem).to_vec()
    }
}

impl WriteSane for u8 {
    fn to_le_bytes(elem: u8) -> Vec<u8> {
        vec![elem]
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

fn write_header<F: Write, A: SaneData, D: Dimension, Repr>(file: &mut F, array: &ArrayBase<Repr, D>)  -> Result<(), WriteError>
where
    Repr: Data<Elem = A>
{
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

fn write_data<F: Write, A: WriteSane, D: Dimension, Repr>(file: &mut F, array: &ArrayBase<Repr, D>) -> Result<(), WriteError>
where
    Repr: Data<Elem = A>
{
    let data_ptr = array.as_ptr();
    let byte_length = array.len() * size_of::<A>();
    if cfg!(endianness = "little") {
        let data_ptr_bytes = data_ptr.cast::<u8>();
        let data_bytes = unsafe { from_raw_parts(data_ptr_bytes, byte_length) };
        file.write_all(data_bytes).map_err(WriteError::Failed)?;
    } else {
        for &elem in array.iter() {
            let elem_bytes = WriteSane::to_le_bytes(elem);
            file.write_all(&elem_bytes).map_err(WriteError::Failed)?;
        }
    }
    Ok(())
}

/// Write array into a SANE-encoded file
pub fn write_sane<F: Write, A: WriteSane, D: Dimension, Repr>(file: &mut F, array: &ArrayBase<Repr, D>) -> Result<(), WriteError>
where
    Repr: Data<Elem = A>
{
    write_header(file, &array)?;
    write_data(file, &array)?;
    Ok(())
}

/// Write array into SANE-encoded file, returning [`std::io::Error`]s
pub fn write_sane_io<F: Write, A: WriteSane, D: Dimension, Repr>(file: &mut F, array: &ArrayBase<Repr, D>) -> Result<(), std::io::Error>
where
    Repr: Data<Elem = A>
{
    write_sane(file, array).map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))
}

/// Write multiple SANE-encoded arrays to a file
pub fn write_sane_arrays<'a, F: Write, A: WriteSane + 'a, D: Dimension + 'a, Arrays, Repr>(
    mut file: F,
    arrays: Arrays,
) -> Result<(), WriteError>
where
    Repr: Data<Elem = A> + 'a,
    Arrays: IntoIterator<Item = &'a ArrayBase<Repr,D>>
{
    for array in arrays.into_iter() {
        write_sane(&mut file, array)?;
    }
    Ok(())
}

/// Write multiple SANE-encoded arrays to a file, returning [`std::io::Error`]s
pub fn write_sane_arrays_io<'a, F: Write, A: WriteSane + 'a, D: Dimension + 'a, Arrays, Repr>(
    mut file: F,
    arrays: Arrays,
) -> Result<(), std::io::Error>
where
    Repr: Data<Elem = A> + 'a,
    Arrays: IntoIterator<Item = &'a ArrayBase<Repr,D>>
{
    for array in arrays.into_iter() {
        write_sane_io(&mut file, array)?;
    }
    Ok(())
}

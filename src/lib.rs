//! Read and write SANE-encoded arrays
//!
//! This is an implementation of the Simple Array of Numbers Encoding (SANE) specification at:
//! <https://github.com/considerate/sane>
pub mod write;
pub mod read;
pub mod data;

#[doc(inline)]
pub use crate::read::{read_sane, read_sane_dyn, read_sane_arrays, read_sane_arrays_dyn, ReadSane};
#[doc(inline)]
pub use crate::write::{write_sane, write_sane_io, write_sane_arrays, write_sane_arrays_io, write_sane_arrays_dyn, WriteSane};
#[doc(inline)]
pub use crate::data::{SaneData, Sane};


#[cfg(test)]
mod tests {
    use ndarray::{Ix2, Array, Ix3};

    use crate::data::Sane;
    use crate::write::{write_sane, write_sane_arrays};
    use crate::read::{read_sane, read_sane_dyn, ParseError, read_sane_arrays};
    use crate::{write_sane_arrays_dyn, read_sane_arrays_dyn};
    extern crate quickcheck;
    use std::io::Cursor;


    #[test]
    fn roundtrip() {
        let arr = ndarray::array![[1,2,3], [4,5,6]];
        let mut file = Cursor::new(Vec::new());
        write_sane(&mut file, &arr).unwrap();
        file.set_position(0);
        let parsed = read_sane_dyn(&mut file).unwrap();
        match parsed {
            Sane::ArrayI32(arr2) => assert_eq!(arr.into_dyn(), arr2),
            _ => assert!(false),
        }
    }

    #[test]
    fn roundtrip_typed() {
        let arr = ndarray::array![[1,2,3], [4,5,6]];
        let mut file = Cursor::new(Vec::new());
        write_sane(&mut file, &arr).unwrap();
        file.set_position(0);
        let arr2 = read_sane(&mut file).unwrap();
        assert_eq!(arr, arr2)
    }

    #[test]
    fn roundtrip_typed_wrong_shape() {
        // This array is rank 2
        let arr = ndarray::array![[1,2,3], [4,5,6]];
        let mut file = Cursor::new(Vec::new());
        write_sane(&mut file, &arr).unwrap();
        file.set_position(0);
        // Parsing as rank 3 should fail with a ShapeError
        let result : Result<Array<i32, Ix3>, _> = read_sane(&mut file);
        match result {
            Err(ParseError::ShapeError(_)) => assert!(true),
            _ => assert!(false),
        }
    }

    #[test]
    fn roundtrip_arrays() {
        let arr = ndarray::array![[1,2,3], [4,5,6]];
        let arr2 = ndarray::array![[7,8], [9,10], [11,12]];
        let mut file = Cursor::new(Vec::new());
        let arrs = [arr, arr2];
        write_sane_arrays(&mut file, &arrs).unwrap();
        file.set_position(0);
        let parsed: Vec<Array<i32, Ix2>> = read_sane_arrays(&mut file).unwrap();
        assert_eq!(parsed, arrs);
    }

    #[test]
    fn roundtrip_hetrogenous_types() {
        use Sane::*;
        let arrs = vec![
            ArrayI32(ndarray::array![[1,2,3], [4,5,-6]].into_dyn()),
            ArrayF32(ndarray::array![[1.0,2.0], [-4.0,5.0]].into_dyn()),
            ArrayF64(ndarray::array![[1.0], [2.0], [3.0], [5.0]].into_dyn()),
            ArrayU8(ndarray::array![[1], [2], [3], [5], [250]].into_dyn()),
            ArrayI8(ndarray::array![[1], [-2], [3], [5], [-128]].into_dyn()),
        ];
        let mut file = Cursor::new(Vec::new());
        write_sane_arrays_dyn(&mut file, &arrs).unwrap();
        file.set_position(0);
        let parsed: Vec<Sane> = read_sane_arrays_dyn(&mut file).unwrap();
        assert_eq!(parsed, arrs)
    }
}

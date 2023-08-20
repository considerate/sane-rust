//! Read and write SANE-encoded arrays
//!
//! This is an implementation of the Simple Array of Numbers Encoding (SANE) specification at:
//! <https://github.com/considerate/sane>
mod write;
mod read;
mod data;

pub use crate::read::{read_sane, read_sane_dyn, ReadSane};
pub use crate::write::{write_sane, WriteSane};
pub use crate::data::SaneData;


#[cfg(test)]
mod tests {
    use crate::data::Sane;
    use crate::write::write_sane;
    use crate::read::{read_sane, read_sane_dyn, ParseError};
    extern crate quickcheck;
    use std::io::Cursor;


    #[test]
    fn example_roundtrip() {
        let arr = ndarray::array![[1,2,3], [4,5,6]];
        let mut file = Cursor::new(Vec::new());
        write_sane(&mut file, arr.clone()).unwrap();
        file.set_position(0);
        let parsed = read_sane_dyn(&mut file).unwrap();
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
        let arr2 = read_sane(&mut file).unwrap();
        assert_eq!(arr, arr2)
    }

    #[test]
    fn example_roundtrip_typed_wrong_shape() {
        let arr = ndarray::array![[1,2,3], [4,5,6]];
        let wrong = ndarray::array![[[1,2,3], [4,5,6]]];
        let mut file = Cursor::new(Vec::new());
        write_sane(&mut file, arr.clone()).unwrap();
        file.set_position(0);
        match read_sane(&mut file) {
            Ok(actual) => assert_ne!(wrong, actual), // This is here to determine the expected type of read_sane
            Err(ParseError::ShapeError(_)) => assert!(true),
            Err(_) => assert!(false),
        }
    }
}

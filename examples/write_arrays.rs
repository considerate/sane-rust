use std::fs::File;
use sane_array::{write_sane, WriteSane};
use ndarray::{self, Dimension};
use ndarray::array;
use std::io::{Error, ErrorKind};

fn write_sane_file<A: WriteSane, D: Dimension>(path: &str, arr: ndarray::Array<A, D>)  -> Result<(), Error> {
    let file = File::create(path)?;
    write_sane(file, arr).map_err(|e| Error::new(ErrorKind::Other, e))
}

fn main() -> std::io::Result<()> {
    write_sane_file("tests/arrays/simple.sane", array![[1,2],[3,4]])?;
    write_sane_file("tests/arrays/scalar.sane", ndarray::arr0(1.0 as f32))?;
    write_sane_file("tests/arrays/vec.sane", ndarray::array![1.0 as f32])?;
    let f64s: ndarray::Array<f64, _> = ndarray::Array::range(1.0, 8.0, 0.5);
    let i8s: ndarray::Array<i8, _> = ndarray::Array::from_iter(-5..5);
    let u8s: ndarray::Array<u8, _> = ndarray::Array::from_iter(0..5);
    write_sane_file("tests/arrays/f64.sane", f64s)?;
    write_sane_file("tests/arrays/i8.sane", i8s)?;
    write_sane_file("tests/arrays/u8.sane", u8s)?;
    let nested = array![
        [[1,2], [3,4], [5,6]],
        [[7,8], [9,10], [11,12]],
        [[13,14], [15,16], [17,18]],
        [[19,20], [21,22], [23,24]],
    ];
    write_sane_file("tests/arrays/nested.sane", nested)?;
    Ok(())
}

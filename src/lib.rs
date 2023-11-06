use pyo3::prelude::*;
use pyo3::types::{IntoPyArray, PyArray1};


mod binning;


#[pyfunction]
fn bin_spectra(
    _py: Python,
    wl_old: Vec<f64>,
    flux_old: Vec<f64>,
    wl_new: Vec<f64>,
) -> PyResult<Vec<f64>> {
    Ok(binning::bin_spectra(wl_old, flux_old, &wl_new))   
}


/// A Python module implemented in Rust.
#[pymodule]
fn _gridpolator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bin_spectra, m)?)?;
    Ok(())
}

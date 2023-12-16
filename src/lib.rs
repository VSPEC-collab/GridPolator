/// The GridPolator binning module in Rust.
/// Benchmarks have shown that this implementation is
/// 20 times faster than Python.
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

mod binning;

#[pymodule]
fn _gridpolator<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn bin_spectra<'py>(
        _py: Python<'py>,
        wl_old: PyReadonlyArray1<'py, f64>,
        flux_old: PyReadonlyArray1<'py, f64>,
        wl_new: PyReadonlyArray1<'py, f64>,
    ) -> &'py PyArray1<f64> {
        let wl_old: Array1<f64> = wl_old.to_owned_array();
        let flux_old: Array1<f64> = flux_old.to_owned_array();
        let wl_new: Array1<f64> = wl_new.to_owned_array();
        let flux_new = binning::bin_spectra(wl_old, flux_old, &wl_new);
        return flux_new.into_pyarray(_py);
    }
    Ok(())
}

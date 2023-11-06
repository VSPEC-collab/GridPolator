/// File for profiling and benchmarking the code.

mod binning;
use binning::bin_spectra;

use std::time::Instant;


fn main() {
    let n_elems = 100000;
    let wl_start = 1.0;
    let wl_end = 20.0;
    let step = (wl_end - wl_start) / n_elems as f64;
    let wl_old = (0..n_elems).map(|x| wl_start + x as f64 * step).collect::<Vec<f64>>();
    let flux_old = vec![1.0;n_elems];
    let n_elems: i32 = 1000;
    let wl_start = 2.0;
    let wl_end = 18.0;
    let step = (wl_end - wl_start) / n_elems as f64;
    let wl_new = (0..n_elems).map(|x| wl_start + x as f64 * step).collect::<Vec<f64>>();
    let start_time = Instant::now();
    let _fl_new = bin_spectra(wl_old, flux_old, &wl_new);
    let elapsed = start_time.elapsed();
    println!("Elapsed time: {:?}", elapsed);
    ()
}   
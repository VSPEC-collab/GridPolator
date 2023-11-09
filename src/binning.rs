/// Bin spectra to a new wavelength grid.
/// This is a rust implementation of the
/// binning algorithm in VSPEC.
/// Benchmarks have shown that this implementation is
/// 50 times faster than Python.
/// 
/// Original algorithm by Cameron Kelahan, Rust
/// implementation and optimizations by Ted Johnson
///


use ndarray::Array1;
/// Bin spectra to a new wavelength grid.
/// 
/// # Arguments
/// * `wl_old` - The old wavelength grid.
/// * `flux_old` - The old flux grid.
/// * `wl_new` - The new wavelength grid.
/// 
/// # Returns
/// * `binned_flux` - The binned flux grid. Note that its length
/// is equal to the length of `wl_new` minus one.
/// 
/// # Notes
/// * This function is a rust implementation of the
/// binning algorithm in VSPEC.
/// * Benchmarks have shown that this implementation is
/// 50 times faster than Python.
/// 
pub fn bin_spectra(
    wl_old: Array1<f64>,
    flux_old: Array1<f64>,
    wl_new: &Array1<f64>,
) -> Array1<f64> {
    
    let new_len = wl_new.len()-1;
    let mut binned_flux: Array1<f64> = Array1::zeros(new_len);
    let mut starting_index = 0;
    for i in 0..new_len {
        // Iterate through each index of the new wavelength grid.
        let lam_cen:f64 = wl_new[i];
        // Get the central wavelength of the bin.
        let upper:f64 = 0.5*(lam_cen+wl_new[i+1]);
        // Get the upper bound of the bin.
        let lower:f64;
        if i == 0 {
            lower = lam_cen
        }
        else {
            lower = 0.5*(lam_cen+wl_new[i-1]);
        }
        // Get the lower bound of the bin.
        // For the first bin, the lower bound is the central wavelength.
        if lower > upper { // Check that nothing weird happened.
            panic!("lower > upper");
        }
        let mut sum:f64 = 0.0;
        let mut num:u32 = 0;
        for (j,wl) in wl_old.iter().enumerate().skip(starting_index) {
            // Iterate through each index of the old wavelength grid.
            // However, since the wavelength grid is sorted, we can
            // skip the first starting_index indices and stop the iteration
            // when the wavelength goes above the upper bound.
            if wl < &lower {
                starting_index = j;
                // Move the starting index when it is below the lower bound.
            }
            else if wl > &upper {
                break;
                // Stop the iteration when the wavelength goes above the upper bound.
            }
            else {
                sum += flux_old[j];
                num += 1;
            }
        }
        if num == 0 { // This might happen if the original grid is of comparible
                      // resolution to the new grid
            panic!("no pixels in bin");
        }
        let mean:f64 = sum/(num as f64);
        // Calculate the mean flux in the bin.
        binned_flux[i] = mean;
        // Store the mean flux in the new grid.
    }
    return binned_flux
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bin_spectra() {
        let wl_old = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
        let flux_old = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let wl_new = Array1::from_vec(vec![0.2, 0.4, 0.6, 0.8]);     
        let binned_flux = bin_spectra(wl_old, flux_old, &wl_new);
        let expected_binned_flux = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        assert_eq!(binned_flux, expected_binned_flux);
    }
}
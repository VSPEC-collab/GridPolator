/// Bin spectra to a new wavelength grid.
/// 

// use std::cmp;
// use std::collections::HashMap;

pub fn bin_spectra(
    wl_old: Vec<f64>,
    flux_old: Vec<f64>,
    wl_new: &Vec<f64>,
) -> Vec<f64> {
    // let mut wl_old_map: HashMap<f64, f64> = HashMap::new();
    // for (wl, flux) in wl_old.iter().zip(flux_old.iter()) {
    //     wl_old_map.insert(wl, flux);
    // }
    let new_len = wl_new.len()-1;
    let mut binned_flux: Vec<f64> = vec![-1.0; new_len];
    let mut starting_index = 0;
    for i in 0..new_len {
        let lam_cen:f64 = wl_new[i];
        let upper:f64 = 0.5*(lam_cen+wl_new[i+1]);
        let lower:f64;
        if i == 0 {
            lower = lam_cen
        }
        else {
            lower = 0.5*(lam_cen+wl_new[i-1]);
        }
        if lower > upper {
            panic!("lower > upper");
        }
        let mut sum:f64 = 0.0;
        let mut num:u32 = 0;
        // println!("Starting index = {}", starting_index);
        for (j,wl) in wl_old.iter().enumerate().skip(starting_index) {
            // println!("{}", starting_index);
            if wl < &lower {
                starting_index = j;
            }
            else if wl > &upper {
                // println!("breaking at {}", j);
                break;
            }
            else {
                sum += flux_old[j];
                num += 1;
            }
        }
        if num == 0 {
            panic!("no pixels in bin");
        }
        let mean:f64 = sum/(num as f64);
        // println!("Placing value {} in bin {}", mean, i);
        binned_flux[i] = mean; //wl_old_map.get(i, mean);
        // println!("binned_flux = {:?}", binned_flux);
    }
    // println!("finally binned_flux = {:?}", binned_flux);
    binned_flux
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bin_spectra() {
        let wl_old = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let flux_old = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let wl_new = vec![0.2, 0.4, 0.6, 0.8];
        let binned_flux = bin_spectra(wl_old, flux_old, &wl_new);
        assert_eq!(binned_flux, vec![1.0, 1.0, 1.0]);
    }
}
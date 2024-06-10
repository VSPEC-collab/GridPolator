// Get the wavelength values of a binned spectrum
// in the same way that PSG does. This ensures that
// things are consistent when working with PSG.

use ndarray::Array1;
// Get the wavelength values of a binned spectrum
// in the same way that PSG does. This ensures that
// things are consistent when working with PSG.
//
// # Arguments
// * `resolving_power` - The resolving power of the spectrum.
// * `lam1` - The first wavelength in the spectrum.
// * `lam2` - The last wavelength in the spectrum.
//
// # Returns
// * `lams` - The wavelength values of the spectrum.
pub fn get_wavelengths(
    resolving_power: f64,
    lam1: f64,
    lam2: f64
) -> Array1<f64> {
    let mut lams = Vec::new();
    lams.push(lam1);

    let mut lam = lam1;
    while lam < lam2 {
        let dlam = lam / resolving_power;
        lam = lam + dlam;
        lams.push(lam);
    }
    Array1::from_shape_vec((lams.len(),), lams).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_get_wavelengths() {
        let lams_high = get_wavelengths(1000.0, 1.0, 2.0);
        let lams_low = get_wavelengths(50.0,1.0,2.0);
        assert_eq!(lams_high[0], 1.0);
        assert_eq!(lams_low[0], 1.0);
        if lams_high.last().unwrap() < &2.0 {
            panic!("lams_high[-1] < 2.0");
        }
        if lams_low.last().unwrap() < &2.0 {
            panic!("lams_low[-1] < 2.0");
        }
        if lams_high.len() <= lams_low.len() {
            panic!("lams_high.len <= lams_low.len");
        }
    }
}
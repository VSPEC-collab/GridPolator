"""
Spectra binning functions
"""

import numpy as np
import warnings

from GridPolator._gridpolator import bin_spectra as bin_spectra_rust


def get_wavelengths(resolving_power: int, lam1: float, lam2: float) -> np.ndarray:
    """
    Get wavelengths

    Get wavelength points given a resolving power and a desired spectral range.
    Provides one more point than PSG, which is alows us to set a red bound on the last pixel.

    Parameters
    ----------
    resolving_power : int
        Resolving power.
    lam1 : float
        Initial wavelength.
    lam2 : float
        Final wavelength.

    Returns
    -------
    numpy.ndarray
        Wavelength points.
    """
    lam = lam1
    lams = [lam]
    while lam < lam2:
        dlam = lam / resolving_power
        lam = lam + dlam
        lams.append(lam)
    lams = np.array(lams)
    return lams



def bin_spectra_python(wl_old: np.ndarray, fl_old: np.ndarray, wl_new: np.ndarray):
    """
    Bin spectra

    This is a generic binning funciton.

    Parameters
    ----------
    wl_old : np.ndarray
        The original wavelength values.
    fl_old : np.ndarray
        The original flux values.
    wl_new : np.ndarray
        The new wavelength values.

    Returns
    -------
    fl_new : np.ndarray
        The new flux values.
    
    Todo
    ----
    Implement in rust or C.
    """
    binned_flux = []
    for i in range(len(wl_new) - 1):
        lam_cen = wl_new[i]
        upper = 0.5*(lam_cen + wl_new[i+1])
        if i == 0:
            # dl = upper - lam_cen # uncomment to sample blue of first pixel
            lower = lam_cen  # - dl
        else:
            lower = 0.5*(lam_cen + wl_new[i-1])
        if lower >= upper:
            raise ValueError('Somehow lower is greater than upper!')
        reg = (wl_old >= lower) & (wl_old < upper)
        if not np.any(reg):
            raise ValueError(
                f'Some pixels must be selected!\nlower={lower}, upper={upper}')
        binned_flux.append(fl_old[reg].mean())
    binned_flux = np.array(binned_flux)
    return binned_flux

def bin_spectra(
    wl_old: np.ndarray,
    fl_old: np.ndarray,
    wl_new: np.ndarray,
    impl:str = 'rust'
):
    """
    Bin Spectra to a new wavelength grid.
    
    Parameters
    ----------
    wl_old : np.ndarray (shape=(N,))
        The original wavelength values.
    fl_old : np.ndarray (shape=(N,))
        The original flux values.
    wl_new : np.ndarray (shape=(M,))
        The new wavelength values.
    impl : str, Optional
        The implementation to use. One of 'rust' or 'python'. Defaults to 'rust'.
    
    Returns
    -------
    np.ndarray (shape=(M-1,))
        The new flux values.
    
    Warns
    -----
    RuntimeWarning
        If `wl_old`, `fl_old`, or `wl_new` are not float64.
    
    Notes
    -----
    The shape of the returned array is one less than the shape of `wl_new`.
    
    """
    def msg(name:str):
        return f'Converting {name} to float64. This may result in degraded performance. Try explicitly specifying the dtype.'
    if not wl_old.dtype == np.float64:
        warnings.warn(msg('wl_old'),RuntimeWarning)
        wl_old = wl_old.astype(np.float64)
    if not fl_old.dtype == np.float64:
        warnings.warn(msg('fl_old'),RuntimeWarning)
        fl_old = fl_old.astype(np.float64)
    if not wl_new.dtype == np.float64:
        warnings.warn(msg('wl_new'),RuntimeWarning)
        wl_new = wl_new.astype(np.float64)
    if impl == 'rust':
        return bin_spectra_rust(wl_old, fl_old, wl_new)
    elif impl == 'python':
        return bin_spectra_python(wl_old, fl_old, wl_new)
    else:
        raise ValueError(f'Unknown implementation: {impl}')
    

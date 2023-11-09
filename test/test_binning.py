"""
Tests for binning functionality.
"""
import numpy as np
from astropy import units as u
import pytest
from time import time


from GridPolator.builtins.phoenix_vspec import RawReader
from GridPolator.binning import get_wavelengths, bin_spectra


def test_get_wavelengths():
    """
    Test for `get_wavelengths()` function
    """
    resolving_power = 1000
    lam1 = 400
    lam2 = 800
    wavelengths = get_wavelengths(resolving_power, lam1, lam2)

    assert isinstance(wavelengths, np.ndarray)
    assert len(wavelengths) > 0
    assert wavelengths[0] == lam1
    # the last pixel gets thrown away after binning
    assert wavelengths[-2] <= lam2
    assert np.all(np.diff(np.diff(wavelengths)) > 0)


@pytest.mark.parametrize(
    "resolving_power, lam1, lam2",
    [
        (1000, 400, 800),
        (500, 350, 600),
        (2000, 600, 1000),
    ],
)
def test_get_wavelengths_parametrized(resolving_power: float, lam1: float, lam2: float):
    """
    Parametrized test for `get_wavelengths()` function
    """
    wavelengths = get_wavelengths(resolving_power, lam1, lam2)

    assert isinstance(wavelengths, np.ndarray)
    assert len(wavelengths) > 0
    assert wavelengths[0] == lam1
    assert wavelengths[-2] <= lam2
    assert np.all(np.diff(np.diff(wavelengths)) > 0)


@pytest.mark.parametrize(
    "wl_old, fl_old, wl_new, expected",
    [
        (
            np.array([400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500], dtype=float),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
            np.array([405, 425, 435, 455, 465, 485], dtype=float),
            np.array([1, 1, 1, 1, 1], dtype=float),
        ),
    ],
)
def test_bin_spectra_parametrized(
    wl_old: np.ndarray,
    fl_old: np.ndarray,
    wl_new: np.ndarray,
    expected: np.ndarray
):
    """
    Parametrized test for `bin_spectra()` function
    """
    binned_flux = bin_spectra(wl_old, fl_old, wl_new, impl='rust')

    assert isinstance(binned_flux, np.ndarray)
    assert len(binned_flux) == len(wl_new) - 1
    assert np.all(binned_flux == expected)


def test_bin_from_phoenix():
    w1 = 1*u.um
    w2 = 18*u.um
    resolving_power = 50
    reader = RawReader()
    wl, fl = reader.read(3000*u.K)
    new_wl = get_wavelengths(resolving_power, w1.value, w2.value)
    start_time = time()
    new_fl_py = bin_spectra(wl.value, fl.value, new_wl, impl='python')
    dtime_py = time() - start_time

    start_time = time()
    new_fl_rs = bin_spectra(wl.value, fl.value, new_wl, impl='rust')
    dtime_rs = time() - start_time

    time_py_over_rs = dtime_py / dtime_rs
    expected_time_py_over_rs = 20
    assert time_py_over_rs > expected_time_py_over_rs, \
        f'Rust binning was only {time_py_over_rs}x faster than Python.'

    assert np.all(np.isclose(new_fl_py, new_fl_rs, atol=1e-6)
                  ), 'Fluxes do not match.'

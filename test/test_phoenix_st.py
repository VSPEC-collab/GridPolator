"""
Tests for ``GridPolator.builtins.phoenix_st``.
"""

from pathlib import Path
from io import BytesIO
import pytest
import numpy as np

from astropy.io import fits
from astropy import units as u

from GridPolator.builtins import phoenix_st
from GridPolator import config


def test_metalicity_str():
    """
    Test the `get_metalicity_str()` function.
    """
    assert phoenix_st.get_metalicity_str(0.5) == 'p05'
    assert phoenix_st.get_metalicity_str(-0.5) == 'm05'
    with pytest.raises(ValueError):
        phoenix_st.get_metalicity_str(1.0)
    with pytest.raises(ValueError):
        phoenix_st.get_metalicity_str(-0.1)


def test_teff_str():
    """
    Test the `get_teff_str()` function.
    """
    assert phoenix_st.get_teff_str(2000) == '2000'
    assert phoenix_st.get_teff_str(70000) == '70000'
    with pytest.raises(ValueError):
        phoenix_st.get_teff_str(1999)
    with pytest.raises(ValueError):
        phoenix_st.get_teff_str(70001)


def test_filename():
    """
    Test the `get_filename()` function.
    """
    assert phoenix_st.get_filename(2000, 0.5) == 'phoenixp05_2000.fits'
    assert phoenix_st.get_filename(70000, -0.5) == 'phoenixm05_70000.fits'


def test_dirname():
    """
    Test the `get_dirname()` function.
    """
    assert phoenix_st.get_dirname(0.5) == 'phoenixp05'
    assert phoenix_st.get_dirname(-0.5) == 'phoenixm05'


def test_url():
    """
    Test the `get_url()` function.
    """
    assert phoenix_st.get_url(
        2000, 0.0) == 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix/phoenixm00/phoenixm00_2000.fits'
    assert phoenix_st.get_url(
        70000, 0.0) == 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix/phoenixm00/phoenixm00_70000.fits'


def test_get_path():
    """
    Test the `get_path()` function.
    """
    assert phoenix_st.get_path(2000, 0.0).as_uri() == (Path.home(
    ) / '.gridpolator' / 'grids' / 'phoenix_st' / 'phoenixm00' / 'phoenixm00_2000.fits').as_uri()
    assert phoenix_st.get_path(70000, 0.0).as_uri() == (Path.home(
    ) / '.gridpolator' / 'grids' / 'phoenix_st' / 'phoenixm00' / 'phoenixm00_70000.fits').as_uri()


def test_get():
    """
    Test the `get()` function.
    """
    dat = phoenix_st.get(2000, 0.0)
    filelike = BytesIO(dat)
    assert isinstance(fits.open(filelike), fits.HDUList)


def test_write():
    """
    Test the `write()` function.
    """
    dat = phoenix_st.get(2000, 0.0)
    phoenix_st.write(2000, 0.0, dat)
    expected_path = Path.home() / '.gridpolator' / 'grids' / 'phoenix_st' / \
        'phoenixm00' / 'phoenixm00_2000.fits'
    assert expected_path.exists()
    expected_path.unlink()


def test_exists():
    """
    Test the `exists()` function.
    """
    assert not phoenix_st.exists(2000, 0.0)
    dat = phoenix_st.get(2000, 0.0)
    phoenix_st.write(2000, 0.0, dat)
    assert phoenix_st.exists(2000, 0.0)
    expected_path = Path.home() / '.gridpolator' / 'grids' / 'phoenix_st' / \
        'phoenixm00' / 'phoenixm00_2000.fits'
    assert expected_path.exists()
    expected_path.unlink()
    assert not phoenix_st.exists(2000, 0.0)


def test_read_fits():
    """
    Test the `read_fits()` function.
    """
    dat = phoenix_st.get(2000, 0.0)
    phoenix_st.write(2000, 0.0, dat)
    assert isinstance(phoenix_st.read_fits(2000, 0.0), fits.HDUList)
    expected_path = Path.home() / '.gridpolator' / 'grids' / 'phoenix_st' / \
        'phoenixm00' / 'phoenixm00_2000.fits'
    assert expected_path.exists()
    expected_path.unlink()
    with pytest.raises(FileNotFoundError):
        phoenix_st.read_fits(2000, 0.0, fail=True)
    assert isinstance(phoenix_st.read_fits(2000, 0.0, fail=False), fits.HDUList)
    expected_path.unlink()

def test_read_raw():
    """
    Test the `read_raw()` function.
    """
    assert isinstance(phoenix_st.read_raw(2000, 0.0, 0.0,fail=False), tuple)
    assert len(phoenix_st.read_raw(2000, 0.0, 0.0)) == 2
    wl, fl = phoenix_st.read_raw(2000, 0.0, 0.0)
    assert isinstance(wl, u.Quantity)
    assert isinstance(fl, u.Quantity)
    wl2, fl2 = phoenix_st.read_raw(2000, 0.0, 1.0)
    assert np.all(wl == wl2)
    assert not np.all(fl == fl2)
    

def test_read():
    """
    Test the `read()` function.
    """
    wl, fl = phoenix_st.read(
        2000,0.0,0.0,400,1*u.um, 2*u.um)
    assert isinstance(wl, u.Quantity)
    assert isinstance(fl, u.Quantity)
    assert len(wl) == len(fl) + 1
    assert wl.unit == config.wl_unit
    assert fl.unit == config.flux_unit
    assert wl[0] == 1*u.um
    assert wl[-1] > 2*u.um
    assert wl[-2] < 2*u.um
    expected_path = Path.home() / '.gridpolator' / 'grids' / 'phoenix_st' / \
        'phoenixm00' / 'phoenixm00_2000.fits'
    assert expected_path.exists()
    expected_path.unlink()


def test_download():
    """
    Test the `download()` function.
    """
    if phoenix_st.exists(2000, 0.0):
        phoenix_st.delete(2000, 0.0)
    assert not phoenix_st.exists(2000, 0.0)
    phoenix_st.download(2000, 0.0)
    assert phoenix_st.exists(2000, 0.0)
    phoenix_st.download(2000, 0.0)
    expected_path = Path.home() / '.gridpolator' / 'grids' / 'phoenix_st' / \
        'phoenixm00' / 'phoenixm00_2000.fits'
    assert expected_path.exists()
    expected_path.unlink()


def test_download_set():
    """
    Test the `download_set()` function.
    """
    teffs = [2000, 2100]
    metalicities = [-0.5]
    phoenix_st.download_set(teffs, metalicities)
    assert phoenix_st.exists(2000, -0.5)
    assert phoenix_st.exists(2100, -0.5)
    expected_path = Path.home() / '.gridpolator' / 'grids' / 'phoenix_st' / \
        'phoenixm05' / 'phoenixm05_2000.fits'
    assert expected_path.exists()
    expected_path.unlink()
    expected_path = Path.home() / '.gridpolator' / 'grids' / 'phoenix_st' / \
        'phoenixm05' / 'phoenixm05_2100.fits'
    assert expected_path.exists()
    expected_path.unlink()

def test_clear():
    """
    Test the `clear()` function.
    """
    if not phoenix_st.exists(2000, 0.0):
        phoenix_st.download(2000, 0.0)
    assert phoenix_st.exists(2000, 0.0)
    phoenix_st.clear()
    assert not phoenix_st.exists(2000, 0.0)

def test_delete():
    """
    Test the `delete()` function.
    """
    if not phoenix_st.exists(2000, 0.0):
        phoenix_st.download(2000, 0.0)
    assert phoenix_st.exists(2000, 0.0)
    phoenix_st.delete(2000, 0.0)
    assert not phoenix_st.exists(2000, 0.0)

if __name__ == '__main__':
    pytest.main(Path(__file__))

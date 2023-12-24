"""
Test the builtin VSPEC PHOENIX grid
"""
from astropy import units as u
import pytest

from GridPolator.builtins import phoenix_vspec
from GridPolator import config

def test_get_filename():
    """
    Test the get_filename function.
    """
    expected = 'lte02300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011.HR.h5'
    assert phoenix_vspec.get_filename(2300) == expected
    with pytest.raises(ValueError):
        phoenix_vspec.get_filename(0)

def test_get_url():
    """
    Test the get_url function.
    """
    expected = 'https://zenodo.org/records/10429325/files/lte02300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011.HR.h5'
    assert phoenix_vspec.get_url(2300) == expected
    with pytest.raises(ValueError):
        phoenix_vspec.get_url(0)

# @pytest.mark.skip()
def test_download():
    """
    Test the download function.
    """
    teff = 2300
    downloaded_originally = phoenix_vspec.is_downloaded(teff)
    path = phoenix_vspec.get_path(teff)
    try:
        if downloaded_originally:
            path.unlink()
        # pylint: disable=protected-access
        phoenix_vspec._download(teff)
        assert phoenix_vspec.is_downloaded(teff)
        assert path.exists()
        path.unlink()
        assert not phoenix_vspec.is_downloaded(teff)
        assert not path.exists()
        phoenix_vspec.download(teff)
        with pytest.raises(ValueError):
            phoenix_vspec.download(teff+1)
        assert phoenix_vspec.is_downloaded(teff)
        assert path.exists()
        if not downloaded_originally:
            path.unlink()
    except Exception as e:
        if downloaded_originally:
            phoenix_vspec.download(teff)
        else:
            if path.exists():
                path.unlink()
            else:
                pass
        raise e

def test_read():
    teff = 2300
    phoenix_vspec.download(teff)
    wl, fl = phoenix_vspec.read(teff)
    assert wl.unit == config.wl_unit
    assert fl.unit == config.flux_unit
    assert len(wl) == len(fl)
    




def test_read_phoenix():
    """
    Test the read_phoenix function.
    """

    teff = 3000
    resolving_power = 100
    w1 = 1 * u.um
    w2 = 2 * u.um
    phoenix_vspec.download(teff)
    wl, fl = phoenix_vspec.read_phoenix(teff, resolving_power, w1, w2)

    assert wl.unit == config.wl_unit
    assert fl.unit == config.flux_unit
    # assert len(wl) == len(fl)

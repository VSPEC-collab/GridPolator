"""
Test the builtin VSPEC PHOENIX grid
"""
from astropy import units as u

from GridPolator.builtins.phoenix_vspec import RawReader, read_phoenix
from GridPolator import config


def test_rawreader():
    """
    Test the RawReader class.
    """
    # pylint: disable=protected-access
    reader = RawReader()
    assert reader._path.exists()
    assert reader._path.is_dir()

    teff = 3000 * u.K
    filename = reader.get_filename(teff)
    assert filename == 'lte03000-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011.HR.h5'
    assert (reader._path / filename).exists()

    wl, fl = reader.read(teff)
    assert wl.unit == config.wl_unit
    assert fl.unit == config.flux_unit
    assert len(wl) == len(fl)


def test_read_phoenix():
    """
    Test the read_phoenix function.
    """

    teff = 3000*u.K
    resolving_power = 100
    w1 = 1 * u.um
    w2 = 2 * u.um
    wl, fl = read_phoenix(teff, resolving_power, w1, w2)

    assert wl.unit == config.wl_unit
    assert fl.unit == config.flux_unit
    # assert len(wl) == len(fl)

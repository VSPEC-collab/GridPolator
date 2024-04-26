"""
Tests for configurations
"""
from astropy import units as u
from GridPolator import config


def test_units():
    """
    Tests that the default units are correct.
    """
    assert config.flux_unit == u.Unit('W m-2 um-1')
    assert config.wl_unit == u.Unit('um')
    assert config.teff_unit == u.Unit('K')
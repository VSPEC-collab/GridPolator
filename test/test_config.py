"""
Tests for configurations
"""

from GridPolator import config


def test_grids_base_dir():
    """
    Tests that the grids base directory exists.
    """
    assert config.GRIDS_BASE_DIR.exists()


def test_vspec_phoenix_dir():
    """
    Tests that the VSPEC PHOENIX directory exists.
    """
    assert config.VSPEC_PHOENIX_DIR.exists()

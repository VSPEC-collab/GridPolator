"""
GridPolator configurations

This module contains global configurations used in the GridPolator and VSPEC codes.
"""
from pathlib import Path
from astropy import units as u

GRIDS_BASE_DIR = Path(__file__).parent / 'grids'

VSPEC_PHOENIX_DIR = GRIDS_BASE_DIR / 'phoenix_vspec'

USER_DATA_PATH = Path.home() / '.gridpolator'
USER_GRIDS_PATH = USER_DATA_PATH / 'grids'


flux_unit = u.Unit('W m-2 um-1')
"""
The standard unit of flux.

This unit is used to standardize the flux in VSPEC calculations.
:math:`W m^{-2} \\mu m^{-1}` is chosen because it is the standard
spectral irrandience unit in PSG.

:type: astropy.units.Unit
"""


teff_unit = u.K
"""
The standard unit of temperature.

This selection standardizes units across the package.

:type: astropy.units.Unit
"""

wl_unit = u.um
"""
The standard unit of wavelength.

The output wavelength can still be changed by the user, but internally
we want units to be consistent.

:type: astropy.units.Unit
"""

planet_distance_unit = u.AU
"""
The standard unit of planetary semimajor axis.

This unit is determined by PSG and used to standardize
the semimajor axis of planets in VSPEC.

:type: astropy.units.Unit
"""

planet_radius_unit = u.km
"""
The standard unit of planet radius.

This unit is determined by PSG and used to standardize
the radius of planets in VSPEC.

:type: astropy.units.Unit
"""

period_unit = u.day
"""
The standard unit of planet orbital period.

This unit is determined by PSG and used to standardize
the orbital and rotational periods of planets in VSPEC.

:type: astropy.units.Unit
"""

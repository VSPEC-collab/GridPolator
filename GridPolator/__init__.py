"""
The ``GridPolator`` Public API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main interaction between the user and ``GridPolator`` is through the
``GridSpectra`` class. The ``GridSpectra`` class is used to store, recall,
and interpolate a grid of spectra. However, since it is implemented in JAX,
it is important to also alert the user of the default units that ``GridSpectra``
uses. The default units are:

* **flux**: ``W m-2 um-1``
* **wavelength**: ``um``
* **temperature**: ``K``
"""

__version__ = '0.3.0'

from GridPolator.grid import GridSpectra

from GridPolator.config import flux_unit, wl_unit, teff_unit
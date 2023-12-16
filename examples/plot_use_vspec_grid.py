"""
Use the VSPEC PHOENIX grid
==========================

This example shows how to use the VSPEC grid.
"""
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt

from GridPolator import GridSpectra
from GridPolator import config

#%%
# Load the PHOENIX grid
# ---------------------
# Load the default VSPEC PHOENIX grid.

wave_short = 1*u.um
wave_long = 10*u.um
resolving_power = 100
teffs = [3000,3100,3200] * u.K

spec = GridSpectra.from_vspec(
    w1=wave_short,
    w2=wave_long,
    resolving_power=resolving_power,
    teffs=teffs
)

#%%
# Recall a spectrum from the grid
# -------------------------------
# ``GridSpectra`` will resample the grid with your supplied
# wavelength array as well as interpolate between :math:`T_{eff}` values.
new_wl:u.Quantity = np.linspace(2,5,40) * u.um
teff = 3050 * u.K

flux = spec.evaluate(new_wl.to_value(config.wl_unit), teff.to_value(config.teff_unit))

plt.plot(new_wl, flux)
plt.xlabel(f'Wavelength ({new_wl.unit:latex})')
_=plt.ylabel(f'Flux ({config.flux_unit:latex})')


"""
Read and bin PHOENIX models
"""
from typing import Tuple

from astropy import units as u
from scipy.interpolate import RegularGridInterpolator
import h5py

from GridPolator import config
from GridPolator.binning import bin_spectra, get_wavelengths

WL_UNIT_NEXTGEN = u.AA
FL_UNIT_NEXGEN = u.Unit('erg cm-2 s-1 cm-1')


class RawReader:
    _path = config.VSPEC_PHOENIX_DIR
    _teff_unit = config.teff_unit
    _wl_unit_model = WL_UNIT_NEXTGEN
    _fl_unit_model = FL_UNIT_NEXGEN

    @staticmethod
    def get_filename(teff: u.Quantity) -> str:
        """
        Get the filename for a raw PHOENIX model.

        Parameters
        ----------
        teff : astropy.units.Quantity
            The effective temperature of the model

        Returns
        -------
        str
            The filename of the model.
        """
        return f'lte0{teff.to_value(config.teff_unit):.0f}-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011.HR.h5'

    def read(self, teff: u.Quantity):
        """
        Read a raw PHOENIX model.

        Parameters
        ----------
        teff : astropy.units.Quantity
            The effective temperature of the model.

        Returns
        -------
        wl : astropy.units.Quantity
            The wavelength axis of the model.
        fl : astropy.units.Quantity
            The flux values of the model.
        """
        fh5 = h5py.File(self._path/self.get_filename(teff), 'r')
        wl = fh5['PHOENIX_SPECTRUM/wl'][()] * self._wl_unit_model
        fl = 10.**fh5['PHOENIX_SPECTRUM/flux'][()] * self._fl_unit_model
        wl = wl.to(config.wl_unit)
        fl = fl.to(config.flux_unit)
        return wl, fl


def read_phoenix(
    teff: u.Quantity,
    resolving_power: float,
    w1: u.Quantity,
    w2: u.Quantity,
    impl: str = 'rust'
) -> Tuple[u.Quantity, u.Quantity]:
    """
    Read a PHOENIX model and return an appropriately binned version

    Parameters
    ----------
    teff : astropy.units.Quantity
        The effective temperature of the model.
    resolving_power : float
        The desired resolving power.
    w1 : astropy.units.Quantity
        The blue wavelength limit.
    w2 : astropy.units.Quantity
        The red wavelenght limit.
    impl : str, Optional
        The binning implementation to use. Defaults to 'rust'.

    Returns
    -------
    wl_new : astropy.units.Quantity
            The wavelength axis of the model.
    fl_new : astropy.units.Quantity
        The flux values of the model.
    """

    wl, flux = RawReader().read(teff)

    wl_new: u.Quantity = get_wavelengths(resolving_power, w1.to_value(
        config.wl_unit), w2.to_value(config.wl_unit))*config.wl_unit
    try:
        fl_new = bin_spectra(
            wl_old=wl.to_value(config.wl_unit),
            fl_old=flux.to_value(config.flux_unit),
            wl_new=wl_new.to_value(config.wl_unit),
            impl=impl
        )*config.flux_unit
    except ValueError:  # if the desired resolving power
        # is close to the original resolving
        # power this might be necessary.
        interp = RegularGridInterpolator(
            points=[wl.to_value(config.wl_unit)],
            values=flux.to_value(config.flux_unit)
        )
        fl_new = interp(wl_new.to_value(config.wl_unit))*config.flux_unit
    return wl_new, fl_new

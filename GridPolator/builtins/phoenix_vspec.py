"""
Read and bin PHOENIX models
"""
from typing import Tuple

from astropy import units as u
from scipy.interpolate import RegularGridInterpolator
import h5py
from pathlib import Path
import requests
from tqdm.auto import tqdm

from GridPolator import config
from GridPolator.binning import bin_spectra, get_wavelengths
from .cache import GRIDS_PATH as BASE_GRIDS_PATH
from ..config import user_agent

WL_UNIT_NEXTGEN = u.AA
FL_UNIT_NEXGEN = u.Unit('erg cm-2 s-1 cm-1')

GRIDS_PATH = BASE_GRIDS_PATH / 'phoenix_vspec'


BASE_URL = 'https://zenodo.org/records/10429325/files/'

ALLOWED_TEFFS = [
    2300,2400,2500,2600,2700,2800,
    2900,3000,3100,3200,3300,3400,
    3500,3600,3700,3800,3900
]



@staticmethod
def get_filename(teff: int) -> str:
    """
    Get the filename for a raw PHOENIX model.

    Parameters
    ----------
    teff : int
        The effective temperature of the model in K.

    Returns
    -------
    str
        The filename of the model.
    
    Raises
    ------
    ValueError
        If the teff is not in ``ALLOWED_TEFFS``.
    """
    if not teff in ALLOWED_TEFFS:
        raise ValueError(f'Invalid teff: {teff}')
    return f'lte{teff:05}-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011.HR.h5'


def get_url(teff:int)->str:
    """
    Get the URL for a PHOENIX model.

    Parameters
    ----------
    teff : int
        The effective temperature of the model in K.

    Returns
    -------
    str
        The URL of the model.
    
    Raises
    ------
    ValueError
        If the teff is not in ``ALLOWED_TEFFS``.
    """
    if not teff in ALLOWED_TEFFS:
        raise ValueError(f'Invalid teff: {teff}')
    return BASE_URL + get_filename(teff)
def get_path(teff:int)->Path:
    """
    Get the path for a PHOENIX model.

    Parameters
    ----------
    teff : int
        The effective temperature of the model in K.

    Returns
    -------
    Path
        The path of the model.
    """
    return GRIDS_PATH / get_filename(teff)

def is_downloaded(teff:int)->bool:
    """
    Check if a PHOENIX model is downloaded.

    Parameters
    ----------
    teff : int
        The effective temperature of the model in K.

    Returns
    -------
    bool
        True if the model is downloaded, False otherwise.
    """
    return get_path(teff).exists()

def _download(teff:int):
    """
    Download a PHOENIX model.

    Parameters
    ----------
    teff : int
        The effective temperature of the model in K.
    """
    url = get_url(teff)
    path = get_path(teff)
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = {'User-Agent': user_agent}
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        total_size = int(response.headers.get('content-length', 0))
        response.close()
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Downloading teff={teff}') as pbar:
            with open(path, 'wb') as file, requests.get(url, headers=headers, stream=True,timeout=30) as response:
                for data in response.iter_content(chunk_size=4096):
                    # Write data to file
                    file.write(data)
                    # Update the progress bar
                    pbar.update(len(data))
    except Exception as e:
        raise RuntimeError(f'Failed to download teff={teff}') from e

def download(teff:int):
    """
    Download a PHOENIX model.

    Parameters
    ----------
    teff : int
        The effective temperature of the model in K.
    """
    if not is_downloaded(teff):
        _download(teff)

def read(teff:int):
    """
    Read a PHOENIX model.

    Parameters
    ----------
    teff : int
        The effective temperature of the model in K.

    Returns
    -------
    wl : astropy.units.Quantity
        The wavelength axis of the model.
    fl : astropy.units.Quantity
        The flux values of the model.
    """
    with h5py.File(get_path(teff), 'r') as fh5:
        wl = fh5['PHOENIX_SPECTRUM/wl'][()] * WL_UNIT_NEXTGEN
        fl = 10.**fh5['PHOENIX_SPECTRUM/flux'][()] * FL_UNIT_NEXGEN
        wl = wl.to(config.wl_unit)
        fl = fl.to(config.flux_unit)
        return wl, fl


def read_phoenix(
    teff: int,
    resolving_power: float,
    w1: u.Quantity,
    w2: u.Quantity,
    impl: str = 'rust'
) -> Tuple[u.Quantity, u.Quantity]:
    """
    Read a PHOENIX model and return an appropriately binned version

    Parameters
    ----------
    teff : int
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

    wl, flux = read(teff)

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

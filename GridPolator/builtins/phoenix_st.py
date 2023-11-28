"""
Grid handling code for PHOENIX spectra hosted by
STScI
"""
from pathlib import Path
import requests
from astropy.io import fits
import warnings

from GridPolator import config

DATA_DIR = config.USER_GRIDS_PATH / 'phoenix_st'

BASE_URL = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix/'

def get_metalicity_str(metalicity: float)->str:
    """
    Convert a metallicity to a string.
    
    Parameters
    ----------
    metallicity : float
        The metallicity.
    
    Returns
    -------
    str
        The metallicity as a string.
    
    Raises
    ------
    ValueError
        If the metallicity is not allowed.
    
    Examples
    --------
    >>> get_metalicity_str(0.5)
    'p05'
    >>> get_metalicity_str(-0.5)
    'm05'
    """
    allowed = [
        0.5,
        0.3,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -2.5,
        -3.0,
        -3.5,
        -4.0,
    ]
    if metalicity not in allowed:
        raise ValueError(f"Metalicity must be one of {allowed}")

    sign_str = 'p' if metalicity > 0. else 'm'
    metalicity = abs(metalicity)
    metalicity = int(metalicity*10)
    return f'{sign_str}{metalicity:02d}'

def get_teff_str(teff:int)->str:
    """
    Convert a teff to a string.
    
    Parameters
    ----------
    teff : int
        The effective temperature in K.
    
    Returns
    -------
    str
        The teff as a string.
    
    Raises
    ------
    ValueError
        If the teff is not allowed.
    
    Examples
    --------
    >>> get_teff_str(5000)
    '5000'
    >>> get_teff_str(70000)
    '70000'
    """
    if teff < 2000:
        raise ValueError("TEFF must be >= 2000 K")
    if teff > 70000:
        raise ValueError("TEFF must be <= 70000 K")
    if not teff % 100 == 0:
        raise ValueError("TEFF must be a multiple of 100")
    teff = int(teff)
    return f'{teff}' # no leading zeros

def get_filename(teff:int, metallicity:float)->str:
    """
    Get the filename for a PHOENIX model.
    
    Parameters
    ----------
    teff : int
        The effective temperature in K.
    metallicity : float
        The metallicity.
    
    Returns
    -------
    str
        The filename of the model.
    
    Examples
    --------
    >>> get_filename(5000, 0.5)
    'phoenixp05_5000.fits'
    """
    return f'phoenix{get_metalicity_str(metallicity)}_{get_teff_str(teff)}.fits'

def get_dirname(metallicity:float)->str:
    """
    Get the subdirectory name for a PHOENIX model.
    
    Parameters
    ----------
    metallicity : float
        The metallicity.
    
    Returns
    -------
    str
        The subdirectory name of the model.
    
    Examples
    --------
    >>> get_dirname(0.5)
    'phoenixp05'
    """
    return f'phoenix{get_metalicity_str(metallicity)}'

def get_url(teff:int, metallicity:float)->str:
    """
    Get the URL hosting a PHOENIX model.
    
    Parameters
    ----------
    teff : int
        The effective temperature in K.
    metallicity : float
        The metallicity.
    
    Returns
    -------
    str
        The URL of the model.
    
    Examples
    --------
    >>> get_url(5000, 0.5)
    'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix/phoenixp05_5000.fits'
    """
    return BASE_URL + get_dirname(metallicity) + '/' + get_filename(teff, metallicity)

def get_path(teff:int, metallicity:float)->Path:
    """
    Get the path to a PHOENIX model.
    
    Parameters
    ----------
    teff : int
        The effective temperature in K.
    metallicity : float
        The metallicity.
    
    Returns
    -------
    pathlib.Path
        The path to the model.
    """
    return DATA_DIR / get_dirname(metallicity) / get_filename(teff, metallicity)

def get(teff:int, metallicity:float)->bytes:
    """
    Get a PHOENIX model from the STScI archive.
    
    Parameters
    ----------
    teff : int
        The effective temperature in K.
    metallicity : float
        The metallicity.
    
    Returns
    -------
    bytes
        The model in FITS format.
    
    Raises
    ------
    FileNotFoundError
        If request fails.
    """
    url = get_url(teff, metallicity)
    result = requests.get(url, timeout=30)
    if result.status_code != 200:
        raise FileNotFoundError(f"Failed to download teff={teff} metallicity={metallicity:.1f}")
    return result.content

def write(teff:int, metallicity:float, data:bytes):
    """
    Write a PHOENIX model to disk.
    
    Parameters
    ----------
    teff : int
        The effective temperature in K.
    metallicity : float
        The metallicity.
    data : bytes
        The model in FITS format.
    """
    path = get_path(teff, metallicity)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

def exists(teff:int, metallicity:float)->bool:
    """
    Check if a PHOENIX model is available in the cache.
    
    Parameters
    ----------
    teff : int
        The effective temperature in K.
    metallicity : float
        The metallicity.
    
    Returns
    -------
    bool
        True if the model is available, False otherwise.
    """
    path = get_path(teff, metallicity)
    return path.exists()

def read(teff:int, metallicity:float, fail=True)->fits.HDUList:
    """
    Read a PHOENIX model from disk.
    
    Parameters
    ----------
    teff : int
        The effective temperature in K.
    metallicity : float
        The metallicity.
    fail : bool
        If True, raise FileNotFoundError if the model is not available.
        If False, download the model if it is not available.
    
    Returns
    -------
    astropy.io.fits.HDUList
        The model in FITS format.
    """
    if not exists(teff, metallicity):
        if fail:
            raise FileNotFoundError
        else:
            dat = get(teff, metallicity)
            write(teff, metallicity, dat)
            del dat
    path = get_path(teff, metallicity)
    return fits.open(path)

def download(teff:int, metallicity:float):
    """
    Download a PHOENIX model from the STScI archive.
    
    If the model is not available in the cache, it will be downloaded.
    
    Parameters
    ----------
    teff : int
        The effective temperature in K.
    metallicity : float
        The metallicity.
    """
    if not exists(teff, metallicity):
        dat = get(teff, metallicity)
        write(teff, metallicity, dat)
        del dat
def download_set(teffs:list, metallicities:list):
    """
    Download a set of PHOENIX models from the STScI archive.
    
    Parameters
    ----------
    teffs : list of int
        The effective temperatures in K.
    metallicities : list of float
        The metallicities.
    """
    for teff in teffs:
        for metallicity in metallicities:
            download(teff, metallicity)

def clear():
    """
    Remove the cache directory.
    """
    for metalicity in DATA_DIR.iterdir():
        if metalicity.is_dir():
            for teff in metalicity.iterdir():
                if teff.is_file():
                    teff.unlink()

def delete(teff:int, metallicity:float):
    """
    Delete a PHOENIX model from the cache.
    
    Parameters
    ----------
    teff : int
        The effective temperature in K.
    metallicity : float
        The metallicity.
    """
    path = get_path(teff, metallicity)
    if path.exists():
        path.unlink()
    else:
        warnings.warn(f"Model not found: {path}") 
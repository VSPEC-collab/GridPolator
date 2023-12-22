from collections import OrderedDict
import numpy as np
from jax import numpy as jnp
from jax import jit
from jax.scipy.interpolate import RegularGridInterpolator
from astropy import units as u
from tqdm.auto import tqdm
from typing import Tuple, List, OrderedDict

from GridPolator.builtins.phoenix_vspec import read_phoenix
from GridPolator.astropy_units import isclose
from GridPolator import config
from GridPolator.binning import bin_spectra


class GridSpectra:
    """
    Store, recall, and interpolate a grid of spectra

    Parameters
    ----------
    wl : astropy.units.Quantity
        The wavelength axis of the spectra.
    spectra : list of numpy.ndarray
        The flux values to place in the grid.
    params : tuple of numpy.ndarray
        The other axes of the grid.

    Examples
    --------
    >>> spectra = [spec1,spec2,spec3]
    >>> params = (3000,3100,3200)
    >>> GridSpectra(wl,spectra,params)

    >>> spectra = [
            [spec11,spec12],
            [spec21,spec22],
            [spec31,spec32]
        ]
    >>> params = [
            (3000,3100,3200), # teff
            (-1,1)          # metalicity
        ]
    >>> GridSpectra(wl,spectra,*params)

    """
    
    def __init__(
        self,
        native_wl: jnp.ndarray,
        params: OrderedDict[str,jnp.ndarray],
        spectra: jnp.ndarray
    ):
        """
        Initialize a grid object.
        
        Parameters
        ----------
        native_wl : jax.numpy.ndarray
            The native wavelength axis of the grid.
        params : OrderedDict
            The other axes of the grid. The order is the same
            as the order of axes in `spectra`.
        spectra : jax.numpy.ndarray
            The flux values to place in the grid. The last dimension
            should be wavelength.
        """
        for param_name, param_val in params.items():
            if not isinstance(param_name,str):
                raise TypeError(f'param_name must be a string, but has type {type(param_name)}.')
            if not isinstance(param_val,jnp.ndarray):
                raise TypeError(f'param_val must be a jax.numpy.ndarray, but has type {type(param_val)}.')
            if param_val.ndim != 1:
                raise ValueError(f'param_val must be 1D, but has shape {param_val.shape}.')
        n_params = len(params)
        if spectra.ndim != n_params + 1:
            raise ValueError(f'spectra must have {n_params} dimensions, but has {spectra.ndim}.')
        for i, (param_name, param_val) in enumerate(params.items()):
            if spectra.shape[i] != len(param_val):
                raise ValueError(f'spectra must have {len(param_val)} values in the {i}th dimension, but has {spectra.shape[i]}.')
        if native_wl.ndim != 1:
            raise ValueError(f'native_wl must be a 1D array, but has shape {native_wl.shape}.')
        wl_len = native_wl.shape[0]
        if spectra.shape[-1] != wl_len:
            raise ValueError(f'spectra must have {native_wl.shape[0]} values in the last dimension, but has {spectra.shape[-1]}.')
        
        param_tup = tuple(params.values())
        interp = [RegularGridInterpolator(param_tup,spectra[...,i]) for i in range(wl_len)]
        
        self._wl = native_wl
        self._interp = interp
        self._params = params
    @staticmethod
    def _evaluate(
        interp: List[RegularGridInterpolator],
        params: Tuple[jnp.ndarray],
        wl_native: jnp.ndarray,
        wl:jnp.ndarray
    ):
        result = jnp.array([_interp(params) for _interp in interp])
        if result.ndim != 2:
            raise ValueError(f'result must have 2 dimensions, but has {result.ndim}.')
        return jnp.array([RegularGridInterpolator((wl_native,),r)(wl) for r in jnp.rollaxis(result,1)])
        

    def evaluate(
            self,
            params: Tuple[jnp.ndarray],
            wl:jnp.ndarray = None
        )->jnp.ndarray:
        """
        Evaluate the grid. `args` has the same order as `params` in the `__init__` method.

        Parameters
        ----------
        wl : astropy.units.Quantity
            The wavelength coordinates to evaluate at.
        args : list of float
            The points on the other axes to evaluate the grid.

        Returns
        -------
        jax.numpy.ndarray
            The flux of the grid at the evaluated points.

        """
        if wl is None:
            wl = self._wl
        if len(params) != len(self._params):
            raise ValueError(f'params must have {len(self._params)} values, but has {len(params)}.')
        for param in params:
            if param.ndim != 1:
                raise ValueError(f'params must be 1D arrays, but has shape {param.shape}.')
        param_lens = jnp.array([param.shape[0] for param in params])
        if not jnp.all(param_lens == param_lens[0]):
            raise ValueError(f'params must have equal lengths, but have lengths {param_lens}.')
        
        return self._evaluate(self._interp,params,self._wl, wl)

    @classmethod
    def from_vspec(
        cls,
        w1: u.Quantity,
        w2: u.Quantity,
        resolving_power: float,
        teffs: u.Quantity,
        impl: str = 'rust'
    ):
        """
        Load the default VSPEC PHOENIX grid.

        Parameters
        ----------
        w1 : astropy.units.Quantity
            The blue wavelength limit.
        w2 : astropy.units.Quantity
            The red wavelength limit.
        resolving_power : float
            The resolving power to use.
        teffs : astropy.units.Quantity
            The temperature coordinates to load.
        impl : str, Optional
            The implementation to use. One of 'rust' or 'python'. Defaults to 'rust'.

        """
        specs = []
        wl = None
        for teff in tqdm(teffs, desc='Loading Spectra', total=len(teffs)):
            wave, flux = read_phoenix(teff, resolving_power, w1, w2, impl=impl)
            specs.append(flux.to_value(config.flux_unit))
            if wl is None:
                wl = wave
            else:
                if not np.all(isclose(wl, wave, 1e-6*u.um)):
                    raise ValueError('Wavelength values are different!')
        params = OrderedDict([('teff', jnp.array([teff.to_value(config.teff_unit) for teff in teffs]))])
        specs = jnp.array(specs)
        return cls(wl[:-1], params, specs)
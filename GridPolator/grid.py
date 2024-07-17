"""
GridPolator.grid
================

The ``GridSpectra`` class stores,
recalls, and interpolates a grid of spectra.
"""
from typing import Tuple, List, Union
from collections import OrderedDict
import numpy as np
from jax import numpy as jnp
from jax import jit
from jax.scipy.interpolate import RegularGridInterpolator as JaxRegularGridInterpolator
from scipy.interpolate import RegularGridInterpolator as ScipyRegularGridInterpolator
from astropy import units as u
from tqdm.auto import tqdm

from .builtins import phoenix_vspec
from .builtins import phoenix_st
from .astropy_units import isclose
from . import config

NDArray = Union[jnp.ndarray, np.ndarray]


class GridSpectra:
    """
    Store, recall, and interpolate a grid of spectra

    Parameters
    ----------
    native_wl : jax.numpy.ndarray or numpy.ndarray
        The native wavelength axis of the grid.
    params : OrderedDict of str and jax.numpy.ndarray or numpy.ndarray
        The other axes of the grid. The order is the same
        as the order of axes in `spectra`.
    spectra : jax.numpy.ndarray or numpy.ndarray
        The flux values to place in the grid. The last dimension
        should be wavelength.
    impl : str, Optional
        The interpolater implementation to use. Either 'scipy' or 'jax'. Defaults to 'scipy'.

    Notes
    -----
    A major question a user might ask is how to chose the implementation. It comes down to balancing
    overhead and performance. The JAX implementation can be noticably faster than Scipy if you plan
    to evaluate many thousands of times. For instances that will probably be evaluated less than
    that (e.g. you are making a phase curve with ~100 epochs) Scipy's lack of overhead is likely
    preferable.

    .. warning::
        If you use the JAX implementation you must make sure all you input arrays
        are `jax.numpy.ndarray`.

    Examples
    --------
    >>> spectra = jnp.array([spec1,spec2,spec3]
    >>> params = {'teff': jnp.array([3000,3100,3200])}
    >>> wl = jnp.linspace(0,10,20)
    >>> GridSpectra(wl,params,spectra)

    >>> spectra = jnp.array([
            [spec11,spec12],
            [spec21,spec22],
            [spec31,spec32]
        ])
    >>> params = {
            'teff': jnp.array([3000,3100,3200]),
            'metalicity': jnp.array([-1,1])
        }
    >>> GridSpectra(wl,params,spectra)

    """

    def __init__(
        self,
        native_wl: NDArray,
        params: OrderedDict[str, NDArray],
        spectra: NDArray,
        impl: str = 'scipy'
    ):
        """
        Initialize a grid object.


        """
        self._obj_interp: Union[ScipyRegularGridInterpolator, JaxRegularGridInterpolator] = {
            'scipy': ScipyRegularGridInterpolator,
            'jax': JaxRegularGridInterpolator
        }[impl]
        for param_name, param_val in params.items():
            if not isinstance(param_name, str):
                raise TypeError(
                    f'param_name must be a string, but has type {type(param_name)}.')
            if impl == 'jax':
                if not isinstance(param_val, jnp.ndarray):
                    raise TypeError(
                        f'param_val must be a jax.numpy.ndarray, but has type {type(param_val)}.')
            if param_val.ndim != 1:
                raise ValueError(
                    f'param_val must be 1D, but has shape {param_val.shape}.')
        n_params = len(params)
        if spectra.ndim != n_params + 1:
            raise ValueError(
                f'spectra must have {n_params + 1} dimensions, but has {spectra.ndim}.')
        for i, (param_name, param_val) in enumerate(params.items()):
            if spectra.shape[i] != len(param_val):
                raise ValueError(
                    f'spectra must have {len(param_val)} values in the {i}th dimension, but has {spectra.shape[i]}.')
        if native_wl.ndim != 1:
            raise ValueError(
                f'native_wl must be a 1D array, but has shape {native_wl.shape}.')
        wl_len = native_wl.shape[0]
        if spectra.shape[-1] != wl_len:
            raise ValueError(
                f'spectra must have {native_wl.shape[0]} values in the last dimension, but has {spectra.shape[-1]}.')

        param_tup = tuple(params.values())
        interp = [self._obj_interp(
            param_tup, spectra[..., i], bounds_error=False, fill_value=None) for i in range(wl_len)]

        self._wl = native_wl
        self._interp = interp
        self._params = params

        def _evaluate(
            interp: List[Union[JaxRegularGridInterpolator, ScipyRegularGridInterpolator]],
            params: Tuple[jnp.ndarray],
            wl_native: jnp.ndarray,
            wl: jnp.ndarray
        ):
            result = jnp.array([_interp(params) for _interp in interp]) if impl == 'jax' else np.array(
                [_interp(params) for _interp in interp]
            )
            if result.ndim != 2:
                raise ValueError(
                    f'result must have 2 dimensions, but has {result.ndim}.')
            if impl == 'jax':
                return jnp.array(
                    [self._obj_interp((wl_native,), r)(wl) for r in jnp.rollaxis(result, 1)]
                    )
            else:
                return np.array(
                    [self._obj_interp((wl_native,), r)(wl) for r in np.rollaxis(result, 1)]
                )
        self._evaluate = jit(_evaluate) if impl == 'jax' else _evaluate

    def evaluate(
        self,
        params: Tuple[NDArray],
        wl: NDArray = None
    ) -> NDArray:
        """
        Evaluate the grid. `args` has the same order as `params` in the `__init__` method.

        Parameters
        ----------
        params : tuple of jax.numpy.ndarray or numpy.ndarray
            The parameter values to evaluate the grid at. They must be in array form,
            even if they are scalars.
        wl : jax.numpy.ndarray or numpy.ndarray, optional
            The wavelength axis to evaluate the grid at. If not provided,
            the native wavelength axis is used.

        Returns
        -------
        jax.numpy.ndarray or numpy.ndarray
            The flux of the grid at the evaluated points.

        Examples
        --------
        >>> grid = GridSpectra(native_wl, params, spectra)
        >>> new_params = (jnp.array([3050, 3100]), jnp.array([0.5, 1])) # Will return two sets of spectra
        >>> grid.evaluate(new_params)

        """
        if wl is None:
            wl = self._wl
        if len(params) != len(self._params):
            raise ValueError(
                f'params must have {len(self._params)} values, but has {len(params)}.')
        for param in params:
            if param.ndim != 1:
                raise ValueError(
                    f'params must be 1D arrays, but has shape {param.shape}.')
        param_lens = jnp.array([param.shape[0] for param in params])
        if not jnp.all(param_lens == param_lens[0]):
            raise ValueError(
                f'params must have equal lengths, but have lengths {param_lens}.')
        try:
            return self._evaluate(self._interp, params, self._wl, wl)
        except ValueError as e:
            msg = 'Double check your native wl range. Remember that the `w2` is not included!'

    @classmethod
    def from_vspec(
        cls,
        w1: u.Quantity,
        w2: u.Quantity,
        resolving_power: float,
        teffs: List[int],
        impl_bin: str = 'rust',
        impl_interp: str = 'scipy',
        fail_on_missing: bool = False
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
        teffs : list of int
            The temperature coordinates to load.
        impl_bin : str, Optional
            The binning implementation to use. Defaults to 'rust'.
        impl_interp : str, Optional
            The interpolation implementation to use. Defaults to 'scipy'.
        fail_on_missing : bool, Optional
            Whether to raise an exception if the grid
            needs to be downloaded. Defaults to `False`.

        """
        specs = []
        wl = None
        _np = {'scipy': np, 'jax': jnp}[impl_interp]
        for teff in tqdm(teffs, desc='Loading Spectra', total=len(teffs)):
            if not phoenix_vspec.is_downloaded(teff):
                if fail_on_missing:
                    raise FileNotFoundError(
                        f'PHOENIX grid for {teff} not found. Set `fail_on_missing` to False to download.')
                else:
                    print(f'PHOENIX grid for {teff} not found. Downloading...')
                    phoenix_vspec.download(teff)
            wave, flux = phoenix_vspec.read_phoenix(
                teff, resolving_power, w1, w2, impl=impl_bin)
            specs.append(flux.to_value(config.flux_unit))
            if wl is None:
                wl = wave
            else:
                if not np.all(isclose(wl, wave, 1e-6*u.um)):
                    raise ValueError('Wavelength values are different!')
        params = OrderedDict(
            [('teff', jnp.array(teffs, dtype=float))])
        specs = _np.array(specs)
        return cls(wl[:-1], params, specs, impl=impl_interp)

    @classmethod
    def from_st(
        cls,
        w1: u.Quantity,
        w2: u.Quantity,
        resolving_power: float,
        teffs: List[int],
        metalicities: List[float],
        loggs: List[float],
        impl_bin: str = 'rust',
        impl_interp: str = 'scipy',
        fail_on_missing: bool = False
    ):
        """
        Load the Grid of Phoenix models from STScI.

        Parameters
        ----------
        w1 : astropy.units.Quantity
            The blue wavelength limit.
        w2 : astropy.units.Quantity
            The red wavelength limit.
        teffs : list of int
            The temperature coordinates to load.
        metalicities : list of float
            The metallicity coordinates to load.
        loggs : list of float
            The logg coordinates to load.
        impl_bin : str, Optional
            The binning implementation to use. Defaults to 'rust'.
        impl_interp : str, Optional
            The interpolation implementation to use. Defaults to 'scipy'.
        fail_on_missing : bool, Optional
            Whether to raise an exception if the grid
            needs to be downloaded. Defaults to false.

        """
        spec3d = []
        _np = {'jax': jnp, 'scipy': np}[impl_interp]
        teff_ax = _np.array(teffs, dtype=float)
        metal_ax = _np.array(metalicities, dtype=float)
        logg_ax = _np.array(loggs, dtype=float)

        for teff in tqdm(teffs, desc='Loading Teff Axis', total=len(teffs)):
            spec2d = []
            for metal in tqdm(metalicities, desc='Loading Metallicity Axis', total=len(metalicities)):
                spec1d = []
                if not phoenix_st.exists(teff, metal):
                    if fail_on_missing:
                        raise FileNotFoundError(
                            f'STScI PHOENIX grid value for {teff} {metal} not found. Set `fail_on_missing` to False to download.')
                    else:
                        print(
                            f'PHOENIX grid for {teff} {metal} not found. Downloading...')
                        phoenix_st.download(teff, metal)
                for logg in tqdm(loggs, desc='Loading Logg Axis', total=len(loggs)):
                    wave, flux = phoenix_st.read(
                        teff, metal, logg, resolving_power, w1, w2, impl=impl_bin)
                    spec1d.append(flux.to_value(config.flux_unit))
                spec2d.append(spec1d)
            spec3d.append(spec2d)

        params = OrderedDict(
            [('teff', teff_ax),
             ('metallicity', metal_ax),
             ('logg', logg_ax)])
        return cls(wave[:-1], params, _np.array(spec3d), impl=impl_interp)

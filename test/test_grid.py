"""
Tests for the `Grid` class
"""

from collections import OrderedDict
import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u

from GridPolator import GridSpectra


def test_grid_1d():
    """
    Test that the GridSpectra class can be initialized
    with 1D arrays.
    """
    wl = jnp.linspace(0, 10, 20)
    params = OrderedDict(
        [
            ('teff', jnp.array([3000, 3100, 3200])),
        ]
    )
    spectra = jnp.asarray(
        [jnp.ones_like(wl)*t for t in params['teff']],
    )
    grid = GridSpectra(wl, params, spectra)
    # pylint: disable-next=protected-access
    first_point = grid._interp[0]
    test_teff = jnp.array([3050])
    assert first_point(test_teff) == test_teff
    assert first_point((test_teff,)) == test_teff


def test_grid_1d_bad_init():
    """
    Test that the GridSpectra class raises an error if the
    parameters are bad.
    """
    wl = jnp.linspace(0, 10, 20)
    params = OrderedDict(
        [
            ('teff', jnp.array([3000, 3100, 3200])),
        ]
    )
    spectra = jnp.asarray(
        [jnp.ones_like(wl)*t for t in params['teff']],
    )

    with pytest.raises(TypeError):
        GridSpectra(wl, OrderedDict(
            [(0, jnp.array([3000, 3100, 3200]))]), spectra)
    with pytest.raises(TypeError):
        GridSpectra(wl, OrderedDict(
            [('teff', np.array([3000, 3100, 3200]))]), spectra)
    with pytest.raises(ValueError):
        GridSpectra(wl, OrderedDict(
            [('teff', jnp.array([[3000, 3100, 3200]]))]), spectra)
    with pytest.raises(ValueError):
        GridSpectra(wl, params, jnp.array([
            [jnp.ones_like(wl) for t in params['teff']],
            [jnp.ones_like(wl) for t in params['teff']],
        ]))
    with pytest.raises(ValueError):
        GridSpectra(jnp.array([wl, wl]), params, spectra)
    with pytest.raises(ValueError):
        GridSpectra(wl, params, jnp.array(
            [jnp.ones_like(wl) for i in range(5)],

        ))


def test_grid_2d():
    """
    Test that the GridSpectra class can be initialized
    with 2D arrays.
    """
    wl = jnp.linspace(0, 10, 20)
    params = OrderedDict(
        [
            ('teff', jnp.array([3000, 3100, 3200])),
            ('metallicity', jnp.array([-1, 0, 1])),
        ]
    )
    spectra = jnp.asarray(
        [
            [jnp.ones_like(wl)*t*m for m in params['metallicity']] for t in params['teff']
        ]
    )
    grid = GridSpectra(wl, params, spectra)
    # pylint: disable-next=protected-access
    first_point = grid._interp[0]
    test_teff = jnp.array([3050, 3000, 3100, 3150])
    test_metallicity = jnp.array([0, 0.5, 1, 0.2])
    assert jnp.all(first_point((test_teff, test_metallicity))
                   == test_teff*test_metallicity)


def test_grid_1d_eval():
    """
    Test that the GridSpectra class can be evaluated
    when created with 1D arrays.
    """
    wl = jnp.linspace(0, 10, 20)
    params = OrderedDict(
        [
            ('teff', jnp.array([3000, 3100, 3200])),
        ]
    )
    spectra = jnp.asarray(
        [jnp.ones_like(wl)*t for t in params['teff']],
    )
    grid = GridSpectra(wl, params, spectra)
    test_teff = jnp.array([3050, 3100])
    result = grid.evaluate((test_teff,))
    for i, t in enumerate(test_teff):
        assert jnp.all(result[i] == t)


def test_grid_eval_2d():
    """
    Test that the GridSpectra class can be evaluated
    when created with 2D arrays.
    """
    wl = jnp.linspace(0, 10, 20)
    params = OrderedDict(
        [
            ('teff', jnp.array([3000, 3100, 3200])),
            ('metallicity', jnp.array([-1, 0, 1])),
        ]
    )
    spectra = jnp.asarray(
        [
            [jnp.ones_like(wl)*t*m for m in params['metallicity']] for t in params['teff']
        ]
    )
    grid = GridSpectra(wl, params, spectra)
    test_teff = jnp.array([3050, 3100])
    test_metalicity = jnp.array([0.5, 1])
    result = grid.evaluate((test_teff, test_metalicity))
    for i, (t, m) in enumerate(zip(test_teff, test_metalicity)):
        assert jnp.all(result[i] == t*m)


def test_vspec():
    """
    Test that the grid can be initialized from
    the vspec grid.
    """
    wl1 = 1*u.um
    wl2 = 10*u.um
    resolving_power = 50
    teffs = [3000, 3100, 3200]

    grid = GridSpectra.from_vspec(
        w1=wl1,
        w2=wl2,
        resolving_power=resolving_power,
        teffs=teffs
    )
    new_wl = jnp.linspace(2, 8, 100)
    flux = grid.evaluate((jnp.array([3050]),), new_wl)
    assert isinstance(flux, jnp.ndarray)
    assert flux.shape == (1, 100)

def test_st_grid():
    """
    Test the STScI PHOENIX grid.
    """
    wl1 = 1*u.um
    wl2 = 10*u.um
    resolving_power = 50
    teffs = [3000, 3100, 3200]
    metalicities = [-2, -1, 0.3]
    loggs = [3, 4, 4.5]
    
    grid = GridSpectra.from_st(
        w1=wl1,
        w2=wl2,
        resolving_power=resolving_power,
        teffs=teffs,
        metalicities=metalicities,
        loggs=loggs,
        impl='rust',
        fail_on_missing=False
    )
    new_wl = jnp.linspace(2, 8, 100)
    flux = grid.evaluate((jnp.array([3050]), jnp.array([0.-0.5]), jnp.array([3.5])), new_wl)
    assert isinstance(flux, jnp.ndarray)
    assert flux.shape == (1, 100)


if __name__ == '__main__':
    pytest.main(args=[__file__])

.. GridPolator documentation master file, created by
   sphinx-quickstart on Tue Nov  7 11:39:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GridPolator's documentation!
=======================================

GridPolator is a library for interpolating grids of stellar spectra.

Read our :doc:`intro` to get started, or see our :doc:`API <api>` for more details.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   intro
   api
   auto_examples/index


About
-----

GridPolator was written by Ashraf Dhahbi and Ted Johnson, and was based on previous interpolation code in the `VSPEC <https://github.com/VSPEC-collab/VSPEC>`_ package.
The interpolation backend can be either
`Scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html#scipy.interpolate.RegularGridInterpolator>`_
or `JAX <https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.interpolate.RegularGridInterpolator.html#jax.scipy.interpolate.RegularGridInterpolator>`_
depending on the needs of the user.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

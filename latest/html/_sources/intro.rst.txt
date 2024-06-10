Introduction
============

``GridPolator`` is a library for interpolating grids of stellar spectra.

Installation
************

To install using pip:

.. code-block:: shell

    pip install gridpolator

or in development mode:

.. code-block:: shell

    git clone https://github.com/VSPEC-collab/GridPolator.git
    cd GridPolator
    pip install -e .

or get a specific release (e.g. `v0.1.0`):

.. code-block:: shell

    pip install git+https://github.com/VSPEC-collab/GridPolator.git@v0.4.0

Getting started
***************

The main class that users interact with is :doc:`GridSpectra <api/GridPolator.GridSpectra>`, which can be initialized many ways.
To make a custom interpolator with your own grid, follow :doc:`Intro to Grids <auto_examples/plot_intro_to_grids>`.
Otherwise, see the :doc:`other examples <auto_examples/index>` to learn about initializing one of the built-in grids.
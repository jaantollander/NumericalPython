Numerical Routines in Python Using Numba and Numpy
==================================================
This guide shows examples on how to

- Write efficient numerical routines in ``Python`` using ``Numba`` and ``Numpy``
- Test numerical code with ``Pytest`` and ``Hypothesis``
- Write clean docstring with `Google style docstrings`_
- Compile API documentation from docstrings with ``Sphinx``

.. _Google style docstrings: http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html


Installation
------------
If you do not have anaconda or miniconda installed you can install it using

::

   ./buildscripts/install_miniconda.sh

After installation export ``miniconda3`` to the path using

::

   export PATH=~/miniconda3/bin:$PATH

and create conda environment to run the code. Replace ``x`` in the script with your desired python version

::

   conda env create python3.x -n name3x -f environment.yml


Tests
-----
Test are implemented using pytest with hypothesis. Test can be run from the commandline using

::

   pytest


Docs
----
Documentation is can be compiled using Sphinx. From ``docs`` directory run command

::

   make html

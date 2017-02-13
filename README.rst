Numerical Routines in Python
============================
This guide shows examples on how to

- Write efficient numerical routines in ``Python`` using ``Numba`` and ``Numpy``
- Test numerical code with ``Pytest`` and ``Hypothesis``
- Write clean docstring with `Google style docstrings`_
- Compile API documentation from docstrings with ``Sphinx``

.. Google style docstrings: http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

Tests
-----
Test uses pytest with hypothesis. From commandline test can be using

::

   pytest


Docs
----
Documentation is compiled using Sphinx. Documentation can be compiled using

::

   make html

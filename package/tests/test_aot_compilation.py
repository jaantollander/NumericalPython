"""Tests for testing ahead-of-time complilation.

Todo:
    - More test functions
"""
from package.aot_compilation import create_extension


def test_aot_compilation():
    def f():
        return True

    create_extension('numba_ext', 'func', 'boolean()', f)

    # We can now import the extension and run it
    from package.numba_ext import func
    assert func()

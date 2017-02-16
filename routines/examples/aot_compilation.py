"""Ahead of time (AOT) compilation of Numba functions

- Exporting module
- Importing module
- Installing AOT compiled modules with setup.py
"""
from numba.pycc import CC


def create_extension(extension_name,
                     exported_name,
                     signature,
                     function):
    """Create Numba pre-compiled module from a function.

    This function is simply an encapsulation of the process of creating
    extensions.

    Args:
        extension_name (str):
        exported_name (str, iterable):
        signature (str, iterable):
        function (str, iterable):
    """
    cc = CC(extension_name)
    cc.verbose = True

    # There could be multiple exports before the compile
    cc.export(exported_name, signature)(function)

    # Finally compile the extension
    cc.compile()

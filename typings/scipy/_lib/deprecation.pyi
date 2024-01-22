"""
This type stub file was generated by pyright.
"""

__all__ = ["_deprecated"]

class _DeprecationHelperStr:
    """
    Helper class used by deprecate_cython_api
    """
    def __init__(self, content, message) -> None: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other) -> bool: ...

def deprecate_cython_api(module, routine_name, new_name=..., message=...):  # -> None:
    """
    Deprecate an exported cdef function in a public Cython API module.

    Only functions can be deprecated; typedefs etc. cannot.

    Parameters
    ----------
    module : module
        Public Cython API module (e.g. scipy.linalg.cython_blas).
    routine_name : str
        Name of the routine to deprecate. May also be a fused-type
        routine (in which case its all specializations are deprecated).
    new_name : str
        New name to include in the deprecation warning message
    message : str
        Additional text in the deprecation warning message

    Examples
    --------
    Usually, this function would be used in the top-level of the
    module ``.pyx`` file:

    >>> from scipy._lib.deprecation import deprecate_cython_api
    >>> import scipy.linalg.cython_blas as mod
    >>> deprecate_cython_api(mod, "dgemm", "dgemm_new",
    ...                      message="Deprecated in Scipy 1.5.0")
    >>> del deprecate_cython_api, mod

    After this, Cython modules that use the deprecated function emit a
    deprecation warning when they are imported.

    """
    ...

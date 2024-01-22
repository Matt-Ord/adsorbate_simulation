"""
This type stub file was generated by pyright.
"""

import contextlib

_config = ...
_cpu_count = ...
_NORM_MAP = ...

@contextlib.contextmanager
def set_workers(workers):  # -> Generator[None, Any, None]:
    """Context manager for the default number of workers used in `scipy.fft`

    Parameters
    ----------
    workers : int
        The default number of workers to use

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fft, signal
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal((128, 64))
    >>> with fft.set_workers(4):
    ...     y = signal.fftconvolve(x, x)

    """
    ...

def get_workers():  # -> Any | int:
    """Returns the default number of workers within the current context

    Examples
    --------
    >>> from scipy import fft
    >>> fft.get_workers()
    1
    >>> with fft.set_workers(4):
    ...     fft.get_workers()
    4
    """
    ...

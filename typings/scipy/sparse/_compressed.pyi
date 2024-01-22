"""
This type stub file was generated by pyright.
"""

from ._data import _data_matrix, _minmax_mixin
from ._index import IndexMixin

"""Base class for sparse matrix formats using compressed storage."""
__all__ = []

class _cs_matrix(_data_matrix, _minmax_mixin, IndexMixin):
    """base matrix class for compressed row- and column-oriented matrices"""
    def __init__(self, arg1, shape=..., dtype=..., copy=...) -> None: ...
    def check_format(self, full_check=...):  # -> None:
        """check whether the matrix format is valid

        Parameters
        ----------
        full_check : bool, optional
            If `True`, rigorous check, O(N) operations. Otherwise
            basic check, O(1) operations (default True).
        """
        ...

    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __lt__(self, other) -> bool: ...
    def __gt__(self, other) -> bool: ...
    def __le__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def multiply(self, other):
        """Point-wise multiplication by another matrix, vector, or
        scalar.
        """
        ...

    def diagonal(self, k=...):  # -> NDArray[Unknown] | NDArray[float64]:
        ...
    def maximum(self, other):  # -> _cs_matrix | NDArray[Any]:
        ...
    def minimum(self, other):  # -> _cs_matrix | NDArray[Any]:
        ...
    def sum(self, axis=..., dtype=..., out=...):  # -> Any | matrix[Any, Any]:
        """Sum the matrix over the given axis.  If the axis is None, sum
        over both rows and columns, returning a scalar.
        """
        ...

    def tocoo(self, copy=...):  # -> coo_array:
        ...
    def toarray(self, order=..., out=...):  # -> NDArray[float64]:
        ...
    def eliminate_zeros(self):  # -> None:
        """Remove zero entries from the matrix

        This is an *in place* operation.
        """
        ...

    has_canonical_format = ...
    def sum_duplicates(self):  # -> None:
        """Eliminate duplicate matrix entries by adding them together

        This is an *in place* operation.
        """
        ...

    has_sorted_indices = ...
    def sorted_indices(self):
        """Return a copy of this matrix with sorted indices"""
        ...

    def sort_indices(self):  # -> None:
        """Sort the indices of this matrix *in place*"""
        ...

    def prune(self):  # -> None:
        """Remove empty space after all non-zero elements."""
        ...

    def resize(self, *shape):  # -> None:
        ...

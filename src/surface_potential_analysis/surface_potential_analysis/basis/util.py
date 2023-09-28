from __future__ import annotations

from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    TypeVarTuple,
    Union,
    Unpack,
    overload,
)

import numpy as np

from .basis_like import AxisVector, BasisLike, BasisWithLengthLike

if TYPE_CHECKING:
    from collections.abc import Iterator

    from surface_potential_analysis.types import (
        ArrayFlatIndexLike,
        ArrayIndexLike,
        ArrayStackedIndexLike,
        FlatIndexLike,
        SingleFlatIndexLike,
        SingleIndexLike,
        SingleStackedIndexLike,
        StackedIndexLike,
    )

    from .stacked_basis import StackedBasisLike


_NF0Inv = TypeVar("_NF0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_ND0Inv = TypeVar("_ND0Inv", bound=int)
_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)
_B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])
_BL0Inv = TypeVar("_BL0Inv", bound=BasisWithLengthLike[Any, Any, Any])
_BL0_co = TypeVar("_BL0_co", bound=BasisWithLengthLike[Any, Any, Any], covariant=True)
_B = TypeVarTuple("_B")
_TS = TypeVarTuple("_TS")


# ruff: noqa: D102
class BasisUtil(BasisLike[Any, Any], Generic[_B0_co]):
    """A class to help with the manipulation of an axis."""

    _basis: _B0_co

    def __init__(self, basis: _B0_co) -> None:
        self._basis = basis

    @property
    def n(self: BasisUtil[BasisLike[Any, _N0Inv]]) -> _N0Inv:
        return self._basis.n

    @property
    def fundamental_n(self: BasisUtil[BasisLike[_NF0Inv, _N0Inv]]) -> _NF0Inv:
        return self._basis.fundamental_n

    @property
    def vectors(
        self: BasisUtil[BasisLike[_NF0Inv, _N0Inv]]
    ) -> np.ndarray[tuple[_N0Inv, _NF0Inv], np.dtype[np.complex_]]:
        return self.__into_fundamental__(np.eye(self.n, self.n))  # type: ignore[return-value]

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        return self._basis.__into_fundamental__(vectors, axis)

    def __from_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        return self._basis.__from_fundamental__(vectors, axis)

    @property
    def nx_points(
        self: BasisUtil[BasisLike[Any, _N0Inv]]
    ) -> np.ndarray[tuple[_N0Inv], np.dtype[np.int_]]:
        return np.arange(0, self.n, dtype=int)  # type: ignore[no-any-return]

    @property
    def nk_points(
        self: BasisUtil[BasisLike[Any, _N0Inv]]
    ) -> np.ndarray[tuple[_N0Inv], np.dtype[np.int_]]:
        return np.fft.ifftshift(  # type: ignore[no-any-return]
            np.arange((-self.n + 1) // 2, (self.n + 1) // 2)
        )

    @property
    def fundamental_nk_points(
        self: BasisUtil[BasisLike[_NF0Inv, Any]]
    ) -> np.ndarray[tuple[_NF0Inv], np.dtype[np.int_]]:
        # We want points from (-self.Nk + 1) // 2 to (self.Nk - 1) // 2
        n = self.fundamental_n
        return np.fft.ifftshift(  # type: ignore[no-any-return]
            np.arange((-n + 1) // 2, (n + 1) // 2)
        )

    @property
    def fundamental_nx_points(
        self: BasisUtil[BasisLike[_NF0Inv, Any]]
    ) -> np.ndarray[tuple[_NF0Inv], np.dtype[np.int_]]:
        return np.arange(  # type: ignore[no-any-return]
            0, self.fundamental_n, dtype=int  # type: ignore[misc]
        )

    @property
    def delta_x(
        self: BasisUtil[BasisWithLengthLike[Any, Any, _ND0Inv]]
    ) -> AxisVector[_ND0Inv]:
        return self._basis.delta_x

    @cached_property
    def dx(
        self: BasisUtil[BasisWithLengthLike[Any, Any, _ND0Inv]]
    ) -> AxisVector[_ND0Inv]:
        return self.delta_x / self.n  # type: ignore[no-any-return, misc]

    @cached_property
    def fundamental_dx(
        self: BasisUtil[BasisWithLengthLike[Any, Any, _ND0Inv]]
    ) -> AxisVector[_ND0Inv]:
        return self.delta_x / self.fundamental_n  # type: ignore[no-any-return,misc]

    @property
    def x_points(
        self: BasisUtil[BasisWithLengthLike[Any, _N0Inv, _ND0Inv]]
    ) -> np.ndarray[tuple[_ND0Inv, _N0Inv], np.dtype[np.int_]]:
        return self.dx[:, np.newaxis] * self.nx_points  # type: ignore[no-any-return]

    @property
    def fundamental_x_points(
        self: BasisUtil[BasisWithLengthLike[_NF0Inv, Any, _ND0Inv]],
    ) -> np.ndarray[tuple[_ND0Inv, _NF0Inv], np.dtype[np.int_]]:
        return self.fundamental_dx[:, np.newaxis] * self.fundamental_nx_points  # type: ignore[no-any-return]

    def __iter__(self: BasisUtil[StackedBasisLike[*_B]]) -> Iterator[Union[*_B]]:
        return self._basis.__iter__()

    @property
    def shape(
        self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]]
    ) -> tuple[int, ...]:
        return self._basis.shape

    @property
    def ndim(self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]]) -> int:
        return self._basis.ndim

    @property
    def fundamental_shape(
        self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]]
    ) -> tuple[int, ...]:
        return self._basis.fundamental_shape

    @property
    def stacked_nk_points(
        self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]],
    ) -> ArrayStackedIndexLike[tuple[int]]:
        nk_mesh = np.meshgrid(
            *[BasisUtil(xi_axis).nk_points for xi_axis in self],
            indexing="ij",
        )
        return tuple(nki.ravel() for nki in nk_mesh)

    @property
    def fundamental_stacked_nk_points(
        self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]]
    ) -> ArrayStackedIndexLike[tuple[int]]:
        nk_mesh = np.meshgrid(
            *[BasisUtil(xi_axis).fundamental_nk_points for xi_axis in self],
            indexing="ij",
        )
        return tuple(nki.ravel() for nki in nk_mesh)

    @property
    def stacked_nx_points(
        self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]]
    ) -> ArrayStackedIndexLike[tuple[int]]:
        nx_mesh = np.meshgrid(
            *[BasisUtil(xi_axis).nx_points for xi_axis in self],
            indexing="ij",
        )
        return tuple(nxi.ravel() for nxi in nx_mesh)

    @property
    def fundamental_stacked_nx_points(
        self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]]
    ) -> ArrayStackedIndexLike[tuple[int]]:
        nx_mesh = np.meshgrid(
            *[BasisUtil(xi_axis).fundamental_nx_points for xi_axis in self],
            indexing="ij",
        )
        return tuple(nxi.ravel() for nxi in nx_mesh)

    @overload
    def get_flat_index(
        self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]],
        idx: SingleStackedIndexLike,
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> np.int_:
        ...

    @overload
    def get_flat_index(
        self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]],
        idx: ArrayStackedIndexLike[Unpack[_TS]],
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> ArrayFlatIndexLike[Unpack[_TS]]:
        ...

    def get_flat_index(
        self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]],
        idx: StackedIndexLike,
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> np.int_ | ArrayFlatIndexLike[Any]:
        """
        Given a stacked index, get the flat index into the Wigner-Seitz cell.

        Parameters
        ----------
        idx : tuple[int, int, int]
            The stacked index
        mode : Literal[&quot;raise&quot;, &quot;wrap&quot;, &quot;clip&quot;], optional
            Specifies how out-of-bounds indices are handled, by default "raise"

        Returns
        -------
        int
            the flattened index into the Wigner-Seitz cell
        """
        return np.ravel_multi_index(idx, self.shape, mode=mode)

    @overload
    def get_stacked_index(
        self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]],
        idx: SingleFlatIndexLike,
    ) -> SingleStackedIndexLike:
        ...

    @overload
    def get_stacked_index(
        self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]],
        idx: ArrayFlatIndexLike[Unpack[_TS]],
    ) -> ArrayStackedIndexLike[Unpack[_TS]]:
        ...

    def get_stacked_index(
        self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]],
        idx: FlatIndexLike,
    ) -> StackedIndexLike:
        """
        Given a flat index, produce a stacked index.

        Parameters
        ----------
        idx : int

        Returns
        -------
        tuple[int, int, int]
        """
        return np.unravel_index(idx, self.shape)

    @overload
    def __getitem__(
        self: BasisUtil[StackedBasisLike[*tuple[_B0Inv, ...]]], index: int
    ) -> BasisLike[Any, Any]:
        ...

    @overload
    def __getitem__(
        self: BasisUtil[StackedBasisLike[*tuple[Any, ...]]], index: slice
    ) -> StackedBasisLike[*tuple[Any, ...]]:
        ...

    def __getitem__(
        self: BasisUtil[StackedBasisLike[*tuple[Any, ...]]], index: int | slice
    ) -> Any:
        return self._basis.__getitem__(index)

    @cached_property
    def volume(self) -> np.float_:
        return np.linalg.det(self.delta_x_stacked)  # type: ignore[no-any-return]

    @cached_property
    def reciprocal_volume(self) -> np.float_:
        return np.linalg.det(self.dk_stacked)  # type: ignore[no-any-return]

    @overload
    def get_k_points_at_index(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]], idx: SingleIndexLike
    ) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        ...

    @overload
    def get_k_points_at_index(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]],
        idx: ArrayIndexLike[Unpack[_TS]],
    ) -> np.ndarray[tuple[int, Unpack[_TS]], np.dtype[np.float_]]:
        ...

    def get_k_points_at_index(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]],
        idx: ArrayIndexLike[Unpack[_TS]] | SingleIndexLike,
    ) -> (
        np.ndarray[tuple[int, Unpack[_TS]], np.dtype[np.float_]]
        | np.ndarray[tuple[int], np.dtype[np.float_]]
    ):
        nk_points = idx if isinstance(idx, tuple) else self.get_stacked_index(idx)
        return np.tensordot(self.dk_stacked, nk_points, axes=(0, 0))  # type: ignore Return type is unknown

    @property
    def k_points(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return self.get_k_points_at_index(self.stacked_nk_points)

    @property
    def fundamental_stacked_k_points(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.tensordot(
            self.fundamental_dk_stacked, self.fundamental_stacked_nk_points, axes=(0, 0)
        )

    @overload
    def get_x_points_at_index(
        self: BasisUtil[StackedBasisLike[*tuple[Any, ...]]], idx: SingleIndexLike
    ) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        ...

    @overload
    def get_x_points_at_index(
        self: BasisUtil[StackedBasisLike[*tuple[Any, ...]]],
        idx: ArrayIndexLike[Unpack[_TS]],
    ) -> np.ndarray[tuple[int, Unpack[_TS]], np.dtype[np.float_]]:
        ...

    def get_x_points_at_index(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0_co, ...]]],
        idx: ArrayIndexLike[Unpack[_TS]] | SingleIndexLike,
    ) -> (
        np.ndarray[tuple[int, Unpack[_TS]], np.dtype[np.float_]]
        | np.ndarray[tuple[int], np.dtype[np.float_]]
    ):
        nx_points = idx if isinstance(idx, tuple) else self.get_stacked_index(idx)
        return np.tensordot(self.dx_stacked, nx_points, axes=(0, 0))  # type: ignore Return type is unknown

    @property
    def x_points_stacked(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return self.get_x_points_at_index(self.stacked_nx_points)

    @property
    def fundamental_x_points_stacked(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.tensordot(
            self.fundamental_dx_stacked, self.fundamental_stacked_nx_points, axes=(0, 0)
        )

    @property
    def delta_x_stacked(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array([axi.delta_x for axi in self])

    @property
    def fundamental_delta_x_stacked(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array([axi.delta_x for axi in self])

    @cached_property
    def dx_stacked(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array([BasisUtil(axi).dx for axi in self])

    @cached_property
    def fundamental_dx_stacked(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array([BasisUtil(axi).fundamental_dx for axi in self])

    @property
    def delta_k_stacked(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array(self.shape)[:, np.newaxis] * self.dk_stacked

    @cached_property
    def fundamental_delta_k_stacked(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array(self.fundamental_shape)[:, np.newaxis] * self.dk_stacked

    @cached_property
    def dk_stacked(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        """Get dk as a list of dk for each axis."""
        return 2 * np.pi * np.linalg.inv(self.delta_x_stacked).T

    @property
    def fundamental_dk_stacked(
        self: BasisUtil[StackedBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return self.dk_stacked

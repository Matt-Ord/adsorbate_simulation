from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis_like import (
    AxisLike,
    AxisVector,
    AxisWithLengthLike,
)
from surface_potential_analysis.util.interpolation import pad_ft_points

_NF0Cov = TypeVar("_NF0Cov", bound=int, covariant=True)
_N0Cov = TypeVar("_N0Cov", bound=int, covariant=True)

_NF0Inv = TypeVar("_NF0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_ND0Inv = TypeVar("_ND0Inv", bound=int)

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
# ruff: noqa: D102


class ExplicitAxis(AxisWithLengthLike[_NF0Cov, _N0Cov, _ND0Inv]):
    """An axis with vectors given as explicit states."""

    def __init__(
        self,
        delta_x: AxisVector[_ND0Inv],
        vectors: np.ndarray[tuple[_N0Cov, _NF0Cov], np.dtype[np.complex_]],
    ) -> None:
        self._delta_x = delta_x
        self._vectors = vectors
        super().__init__()

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x

    @property
    def n(self) -> _N0Cov:
        return self.vectors.shape[0]  # type: ignore[no-any-return]

    @property
    def fundamental_n(self) -> _NF0Cov:
        return self.vectors.shape[1]  # type: ignore[no-any-return]

    @property
    def vectors(self) -> np.ndarray[tuple[_N0Cov, _NF0Cov], np.dtype[np.complex_]]:
        return self._vectors

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        transformed = np.tensordot(vectors, self.vectors, axes=([axis], [0]))
        return np.moveaxis(transformed, -1, axis)  # type: ignore[no-any-return]

    @classmethod
    def from_momentum_vectors(
        cls: type[ExplicitAxis[_NF0Cov, _N0Cov, _ND0Inv]],
        delta_x: AxisVector[_ND0Inv],
        vectors: np.ndarray[tuple[_N0Cov, _NF0Cov], np.dtype[np.complex_]],
    ) -> ExplicitAxis[_NF0Cov, _N0Cov, _ND0Inv]:
        vectors = np.fft.ifft(vectors, axis=1, norm="ortho")
        return cls(delta_x, vectors)


class ExplicitAxis1d(ExplicitAxis[_NF0Inv, _N0Inv, Literal[1]]):
    """An axis with vectors given as explicit states with a 1d basis vector."""


class ExplicitAxis2d(ExplicitAxis[_NF0Inv, _N0Inv, Literal[2]]):
    """An axis with vectors given as explicit states with a 2d basis vector."""


class ExplicitAxis3d(ExplicitAxis[_NF0Inv, _N0Inv, Literal[3]]):
    """An axis with vectors given as explicit states with a 3d basis vector."""


class FundamentalAxis(AxisLike[_NF0Cov, _NF0Cov]):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, n: _NF0Cov) -> None:
        self._n = n
        super().__init__()

    @property
    def n(self) -> _NF0Cov:
        return self._n

    @property
    def fundamental_n(self) -> _NF0Cov:
        return self._n

    def __into_fundamental__(  # type: ignore[override]
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        return vectors.astype(np.complex_, copy=False)  # type: ignore[no-any-return]

    def __from_fundamental__(  # type: ignore[override]
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        return vectors.astype(np.complex_, copy=False)  # type: ignore[no-any-return]


class FundamentalPositionAxis(
    FundamentalAxis[_NF0Cov], AxisWithLengthLike[_NF0Cov, _NF0Cov, _ND0Inv]
):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, delta_x: AxisVector[_ND0Inv], n: _NF0Cov) -> None:
        self._delta_x = delta_x
        super().__init__(n)

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x


class FundamentalPositionAxis1d(FundamentalPositionAxis[_NF0Inv, Literal[1]]):
    """A axis with vectors that are the fundamental position states with a 1d basis vector."""


class FundamentalPositionAxis2d(FundamentalPositionAxis[_NF0Inv, Literal[2]]):
    """A axis with vectors that are the fundamental position states with a 2d basis vector."""


class FundamentalPositionAxis3d(FundamentalPositionAxis[_NF0Inv, Literal[3]]):
    """A axis with vectors that are the fundamental position states with a 3d basis vector."""


class MomentumAxis(AxisWithLengthLike[_NF0Cov, _N0Cov, _ND0Inv]):
    """A axis with vectors which are the n lowest frequency momentum states."""

    def __init__(
        self, delta_x: AxisVector[_ND0Inv], n: _N0Cov, fundamental_n: _NF0Cov
    ) -> None:
        self._delta_x = delta_x
        self._n = n
        self._fundamental_n = fundamental_n
        assert self._fundamental_n >= self.n
        super().__init__()

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x

    @property
    def n(self) -> _N0Cov:
        return self._n

    @property
    def fundamental_n(self) -> _NF0Cov:
        return self._fundamental_n

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        padded = pad_ft_points(vectors, s=(self.fundamental_n,), axes=(axis,))
        return np.fft.ifft(padded, self.fundamental_n, axis=axis, norm="ortho")  # type: ignore[no-any-return]

    def __from_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        transformed = np.fft.fft(vectors, self.fundamental_n, axis=axis, norm="ortho")
        return pad_ft_points(transformed, s=(self.n,), axes=(axis,))


class MomentumAxis1d(MomentumAxis[_NF0Inv, _N0Inv, Literal[1]]):
    """A axis with vectors which are the n lowest frequency momentum states with a 1d basis vector."""


class MomentumAxis2d(MomentumAxis[_NF0Inv, _N0Inv, Literal[2]]):
    """A axis with vectors which are the n lowest frequency momentum states with a 2d basis vector."""


class MomentumAxis3d(MomentumAxis[_NF0Inv, _N0Inv, Literal[3]]):
    """A axis with vectors which are the n lowest frequency momentum states with a 3d basis vector."""


class FundamentalMomentumAxis(MomentumAxis[_NF0Cov, _NF0Cov, _ND0Inv]):
    """An axis with vectors which are the fundamental momentum states."""

    def __init__(self, delta_x: AxisVector[_ND0Inv], n: _NF0Cov) -> None:
        super().__init__(delta_x, n, n)

    @property
    def vectors(self) -> np.ndarray[tuple[_NF0Cov, _NF0Cov], np.dtype[np.complex_]]:
        all_states_in_k = np.eye(self.fundamental_n, self.fundamental_n)
        return np.fft.ifft(all_states_in_k, axis=1, norm="ortho")  # type: ignore[no-any-return]


class FundamentalMomentumAxis1d(FundamentalMomentumAxis[_NF0Inv, Literal[1]]):
    """An axis with vectors which are the fundamental momentum states with a 1d basis vector."""


class FundamentalMomentumAxis2d(FundamentalMomentumAxis[_NF0Inv, Literal[2]]):
    """An axis with vectors which are the fundamental momentum states with a 2d basis vector."""


class FundamentalMomentumAxis3d(FundamentalMomentumAxis[_NF0Inv, Literal[3]]):
    """An axis with vectors which are the fundamental momentum states with a 3d basis vector."""

from __future__ import annotations

import functools
import hashlib
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
    cast,
    override,
)

import numpy as np
from scipy.optimize import curve_fit  # type: ignore unknown
from slate.array import Array
from slate.basis import FundamentalBasis, as_index_basis
from slate.metadata import LabeledMetadata

from adsorbate_simulation.system import SimulationCondition

if TYPE_CHECKING:
    from collections.abc import Callable


class FitMethod[T, I: Any](ABC):
    """A method used for fitting an ISF."""

    @override
    def __hash__(self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(self.get_fit_label().encode())
        return int.from_bytes(h.digest(), "big")

    @staticmethod
    @abstractmethod
    def _fit_fn(
        x: np.ndarray[Any, np.dtype[np.floating]],
        *params: *tuple[float, ...],
    ) -> np.ndarray[Any, np.dtype[np.floating]]: ...

    @staticmethod
    @abstractmethod
    def _params_from_fit(
        fit: T,
    ) -> tuple[float, ...]: ...

    @staticmethod
    @abstractmethod
    def _fit_from_params(
        *params: *tuple[float, ...],
    ) -> T: ...

    @staticmethod
    @abstractmethod
    def _scale_params(
        dx: float,
        params: tuple[float, ...],
    ) -> tuple[float, ...]: ...

    @staticmethod
    def _scale_y_data(
        data: np.ndarray[Any, np.dtype[np.floating[Any]]],
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        return data

    @classmethod
    def _scale_y_error(
        cls,
        data: np.ndarray[Any, np.dtype[np.floating[Any]]],
        y_error: np.ndarray[Any, np.dtype[np.floating[Any]]],
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        return cls._scale_y_data(y_error + data) - cls._scale_y_data(data)

    @staticmethod
    @abstractmethod
    def _fit_param_bounds() -> tuple[list[float], list[float]]: ...

    @abstractmethod
    def _fit_param_initial_guess(
        self, data: Array[LabeledMetadata[np.floating], np.floating], info: I
    ) -> tuple[float, ...]: ...

    @abstractmethod
    def get_fit_label(self) -> str: ...

    def get_fit_from_data(
        self,
        data: Array[LabeledMetadata[np.floating], np.floating],
        info: I,
        *,
        y_error: Array[LabeledMetadata[np.floating], np.floating] | None = None,
    ) -> T:
        converted = data.with_basis(as_index_basis(data.basis))
        y_data = converted.raw_data

        x_values = np.asarray(data.basis.metadata().values)[converted.basis.points]
        delta_x = np.max(x_values) - np.min(x_values)
        dx = (delta_x / x_values.size).item()

        sigma = (
            self._scale_y_error(y_data, y_error.with_basis(converted.basis).raw_data)
            if y_error is not None
            else None
        )

        def _fit_fn(
            x: np.ndarray[Any, np.dtype[np.floating]], *params: *tuple[float, ...]
        ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
            return self._scale_y_data(self._fit_fn(x, *params))

        parameters, _covariance = cast(
            "tuple[list[float], Any]",
            curve_fit(
                _fit_fn,
                x_values / dx,
                self._scale_y_data(y_data),
                p0=self._scale_params(
                    dx,
                    self._fit_param_initial_guess(data, info),
                ),
                bounds=self._fit_param_bounds(),
                sigma=sigma,
            ),
        )

        return self._fit_from_params(*self._scale_params(1 / dx, tuple(parameters)))

    @classmethod
    def get_fitted_data[M: LabeledMetadata[np.floating]](
        cls: type[Self],
        fit: T,
        times: M,
    ) -> Array[M, np.floating]:
        data = cls._fit_fn(np.asarray(times.values), *cls._params_from_fit(fit))
        return Array(FundamentalBasis(times), data)

    @classmethod
    def get_function_for_fit[M: LabeledMetadata[np.floating]](
        cls: type[Self],
        fit: T,
    ) -> Callable[[M], Array[M, np.floating]]:
        return functools.partial(cls.get_fitted_data, fit)

    @classmethod
    def n_params(cls: type[Self]) -> int:
        return len(cls._fit_param_bounds()[0])


class ISFFitMethod[T](FitMethod[T, SimulationCondition]):
    def get_rate_from_data(
        self,
        data: Array[LabeledMetadata[np.floating], np.floating],
        info: SimulationCondition,
    ) -> float:
        fit = self.get_fit_from_data(data, info)
        return self.get_rate_from_fit(fit)

    @abstractmethod
    def get_rate_from_fit(
        self,
        fit: T,
    ) -> float: ...

    @abstractmethod
    def get_fit_times(
        self, info: SimulationCondition
    ) -> LabeledMetadata[np.floating]: ...

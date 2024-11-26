from __future__ import annotations

import functools
import hashlib
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Self,
    cast,
    override,
)

import numpy as np
from scipy.optimize import curve_fit  # type: ignore unknown
from slate.array import SlateArray
from slate.basis import FundamentalBasis
from slate.metadata import LabeledMetadata

from adsorbate_simulation.system import SimulationCondition, SimulationPotential


class FitMethod[T, I: Any](ABC):
    """A method used for fitting an ISF."""

    @override
    def __hash__(self: Self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(self.get_rate_label().encode())
        return int.from_bytes(h.digest(), "big")

    @abstractmethod
    def get_rate_from_fit(
        self: Self,
        fit: T,
    ) -> float: ...

    @staticmethod
    @abstractmethod
    def _fit_fn(
        x: np.ndarray[Any, np.dtype[np.float64]],
        *params: *tuple[float, ...],
    ) -> np.ndarray[Any, np.dtype[np.float64]]: ...

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
        dt: float,
        params: tuple[float, ...],
    ) -> tuple[float, ...]: ...

    @staticmethod
    @abstractmethod
    def _fit_param_bounds() -> tuple[list[float], list[float]]: ...

    @abstractmethod
    def _fit_param_initial_guess(
        self: Self, data: SlateArray[LabeledMetadata[float], np.float64], info: I
    ) -> tuple[float, ...]: ...

    @abstractmethod
    def get_rate_label(self: Self) -> str: ...

    @abstractmethod
    def get_fit_times(self: Self, info: I) -> LabeledMetadata[float]: ...

    def get_fit_from_isf(
        self: Self, data: SlateArray[LabeledMetadata[float], np.float64], info: I
    ) -> T:
        y_data = data.raw_data
        times = np.asarray(data.basis.metadata().values)
        delta_t = np.max(times) - np.min(times)
        dt = (delta_t / times.size).item()

        def _fit_fn(
            x: np.ndarray[Any, np.dtype[np.float64]],
            *params: *tuple[float, ...],
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
            return np.real(self._fit_fn(x, *params))

        parameters, _covariance = cast(
            tuple[list[float], Any],
            curve_fit(
                _fit_fn,
                times / dt,
                y_data,
                p0=self._scale_params(
                    1 / dt,
                    self._fit_param_initial_guess(data, info),
                ),
                bounds=self._fit_param_bounds(),
            ),
        )

        return self._fit_from_params(*self._scale_params(dt, tuple(parameters)))

    def get_rate_from_isf(
        self: Self, data: SlateArray[LabeledMetadata[float], np.float64], info: I
    ) -> float:
        fit = self.get_fit_from_isf(data, info)
        return self.get_rate_from_fit(fit)

    @classmethod
    def get_fitted_data[M: LabeledMetadata[float]](
        cls: type[Self],
        fit: T,
        times: M,
    ) -> SlateArray[M, np.float64]:
        data = cls._fit_fn(np.asarray(times.values), *cls._params_from_fit(fit))
        return SlateArray(FundamentalBasis(times), data)

    @classmethod
    def get_function_for_fit[M: LabeledMetadata[float]](
        cls: type[Self],
        fit: T,
    ) -> Callable[[M], SlateArray[M, np.float64]]:
        return functools.partial(cls.get_fitted_data, fit)

    @classmethod
    def n_params(cls: type[Self]) -> int:
        return len(cls._fit_param_bounds()[0])


class ISFFitMethod[T](FitMethod[T, SimulationCondition[SimulationPotential]]): ...

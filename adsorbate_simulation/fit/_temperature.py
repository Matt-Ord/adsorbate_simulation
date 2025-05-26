from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import numpy as np
from slate_core import Array, FundamentalBasis, array
from slate_core.metadata import ExplicitLabeledMetadata

from adsorbate_simulation.fit._fit_method import FitData, FitMethod
from adsorbate_simulation.util import get_thermal_occupation


@dataclass(frozen=True)
class TemperatureFitInfo:
    target_temperature: float


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float


class TemperatureFitMethod(FitMethod[TemperatureFit, TemperatureFitInfo]):
    NOISE_THRESHOLD = 1e-5

    @staticmethod
    @override
    def _fit_fn(
        x: np.ndarray[Any, np.dtype[np.floating]],
        *params: *tuple[float, ...],
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        (temperature,) = params
        return get_thermal_occupation(x, temperature)

    @staticmethod
    @override
    def _params_from_fit(
        fit: TemperatureFit,
    ) -> tuple[float, ...]:
        return (fit.temperature,)

    @staticmethod
    @override
    def _fit_from_params(
        *params: *tuple[float, ...],
    ) -> TemperatureFit:
        (temperature,) = params
        return TemperatureFit(temperature)

    @staticmethod
    @override
    def _scale_params(
        dx: float,
        params: tuple[float, ...],
    ) -> tuple[float, ...]:
        (temperature,) = params
        return (temperature / dx,)

    @staticmethod
    @override
    def _scale_y_data(
        data: np.ndarray[Any, np.dtype[np.floating[Any]]],
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        return np.log(data)

    @classmethod
    @override
    def _scale_y_error(
        cls,
        data: np.ndarray[Any, np.dtype[np.floating[Any]]],
        y_error: np.ndarray[Any, np.dtype[np.floating[Any]]],
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        return y_error / data

    @override
    def get_fit_from_data(
        self,
        data: FitData,
        info: TemperatureFitInfo,
        *,
        y_error: FitData | None = None,
    ) -> TemperatureFit:
        data = array.as_index_basis(data)
        energies = data.basis.metadata().values[data.basis.points]
        target_occupation = get_thermal_occupation(energies, info.target_temperature)
        above_noise = target_occupation > self.NOISE_THRESHOLD

        truncated = Array(
            FundamentalBasis(ExplicitLabeledMetadata(energies[above_noise])),
            data.raw_data[above_noise],
        )
        truncated_error = (
            Array(
                FundamentalBasis(ExplicitLabeledMetadata(energies[above_noise])),
                y_error.with_basis(data.basis).raw_data[above_noise],
            )
            if y_error is not None
            else None
        )
        return super().get_fit_from_data(truncated, info, y_error=truncated_error)

    @staticmethod
    @override
    def _fit_param_bounds() -> tuple[list[float], list[float]]:
        return ([0.0], [np.inf])

    @override
    def _fit_param_initial_guess(
        self,
        data: FitData,
        info: TemperatureFitInfo,
    ) -> tuple[float, ...]:
        return (info.target_temperature,)

    @override
    def get_fit_label(self) -> str:
        return "temperature"

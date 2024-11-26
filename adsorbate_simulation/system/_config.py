from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self, override

import numpy as np

if TYPE_CHECKING:
    from adsorbate_simulation.system._basis import SimulationBasis

_DEFAULT_DIRECTION = ()


class Environment: ...


class ClosedEnvironment(Environment):
    @override
    def __eq__(self, value: object) -> bool:
        return isinstance(value, ClosedEnvironment)

    @override
    def __hash__(self) -> int:
        return 0


@dataclass(frozen=True, kw_only=True)
class SimulationConfig:
    """Configure the simlation-specific detail of the system."""

    simulation_basis: SimulationBasis
    environment: Environment = field(default_factory=ClosedEnvironment, kw_only=True)
    temperature: float = field(default=150, kw_only=True)
    scattered_energy_range: tuple[float, float] = field(
        default=(-np.inf, np.inf),
        kw_only=True,
    )
    direction: tuple[int, ...] = field(default=_DEFAULT_DIRECTION, kw_only=True)

    def __post_init__(self: Self) -> None:
        assert self.scattered_energy_range[0] <= self.scattered_energy_range[1]
        if self.direction is _DEFAULT_DIRECTION:
            self.direction = tuple(0 for _ in self.simulation_basis.shape)  # type: ignore frozen

    def with_simulation_basis(self: Self, simulation_basis: SimulationBasis) -> Self:
        """Create a new config with different simulation basis."""
        return type(self)(
            simulation_basis=simulation_basis,
            environment=self.environment,
            temperature=self.temperature,
            scattered_energy_range=self.scattered_energy_range,
            direction=self.direction,
        )

    def with_environment(self: Self, environment: Environment) -> Self:
        """Create a new config with different environment."""
        return type(self)(
            simulation_basis=self.simulation_basis,
            environment=environment,
            temperature=self.temperature,
            scattered_energy_range=self.scattered_energy_range,
            direction=self.direction,
        )

    def with_temperature(self: Self, temperature: float) -> Self:
        """Create a new config with different temperature."""
        return type(self)(
            simulation_basis=self.simulation_basis,
            environment=self.environment,
            temperature=temperature,
            scattered_energy_range=self.scattered_energy_range,
            direction=self.direction,
        )

    def with_scattered_energy_range(
        self: Self,
        energy_range: tuple[float, float],
    ) -> Self:
        """Create a new config with different scattered energy range."""
        return type(self)(
            simulation_basis=self.simulation_basis,
            environment=self.environment,
            temperature=self.temperature,
            scattered_energy_range=energy_range,
            direction=self.direction,
        )

    def with_direction(self: Self, direction: tuple[int, ...]) -> Self:
        """Create a new config with different direction."""
        return type(self)(
            simulation_basis=self.simulation_basis,
            environment=self.environment,
            temperature=self.temperature,
            scattered_energy_range=self.scattered_energy_range,
            direction=direction,
        )

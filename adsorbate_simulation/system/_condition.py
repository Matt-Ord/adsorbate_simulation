from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

from adsorbate_simulation.system._config import IsotropicSimulationConfig

if TYPE_CHECKING:
    import numpy as np
    from slate.metadata import (
        AxisDirections,
        SpacedLengthMetadata,
        SpacedVolumeMetadata,
    )
    from slate_quantum.noise import (
        DiagonalNoiseOperatorList,
        NoiseOperatorList,
    )
    from slate_quantum.operator import Operator, Potential

    from adsorbate_simulation.system._config import SimulationConfig
    from adsorbate_simulation.system._system import System


@dataclass(frozen=True)
class SimulationCondition[
    S: System[Any] = System[Any],
    C: SimulationConfig = SimulationConfig,
]:
    system: S
    config: C

    def with_system[_S: System[Any]](self, system: _S) -> SimulationCondition[_S, C]:
        """Create a new condition with different system."""
        return SimulationCondition(system, self.config)

    def with_config[_C: SimulationConfig](
        self, config: _C
    ) -> SimulationCondition[S, _C]:
        """Create a new condition with different config."""
        return SimulationCondition(self.system, config)

    def with_temperature(self, temperature: float) -> SimulationCondition[S, C]:
        """Create a new condition with different temperature."""
        return self.with_config(self.config.with_temperature(temperature))

    @property
    def potential(
        self,
    ) -> Potential[SpacedLengthMetadata, AxisDirections, np.complexfloating]:
        """Get the potential for the simulation."""
        return self.system.get_potential(self.config.simulation_basis)

    @property
    def hamiltonian(
        self,
    ) -> Operator[SpacedVolumeMetadata, np.complexfloating]:
        return self.system.get_hamiltonian(self.config.simulation_basis)

    @property
    def fundamental_metadata(self) -> SpacedVolumeMetadata:
        return self.config.simulation_basis.get_fundamental_metadata(self.system.cell)

    @overload
    def get_environment_operators[_C: IsotropicSimulationConfig](
        self: SimulationCondition[Any, _C],
    ) -> DiagonalNoiseOperatorList[SpacedVolumeMetadata]: ...
    @overload
    def get_environment_operators(
        self,
    ) -> NoiseOperatorList[SpacedVolumeMetadata]: ...

    def get_environment_operators(
        self,
    ) -> NoiseOperatorList[SpacedVolumeMetadata]:
        metadata = self.fundamental_metadata
        return self.config.environment.get_operators(metadata)

    @property
    def temperature_corrected_operators(
        self,
    ) -> NoiseOperatorList[SpacedVolumeMetadata]:
        return self.config.get_temperature_corrected_operators(self.hamiltonian)

    @property
    def temperature(self) -> float:
        """The temperature of the simulation."""
        return self.config.temperature

    @property
    def direction(self) -> tuple[int, ...]:
        """The direction of the scattering of the simulation."""
        return self.config.direction or tuple(0 for _ in self.system.cell.lengths)

    @property
    def scattered_energy_range(self) -> tuple[float, float]:
        """The scattered energy range of the simulation."""
        return self.config.scattered_energy_range

    @property
    def mass(self) -> float:
        """The mass of the system."""
        return self.system.mass

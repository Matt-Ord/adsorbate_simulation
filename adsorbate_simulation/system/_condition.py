from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Never, overload

from adsorbate_simulation.system._config import IsotropicSimulationConfig

if TYPE_CHECKING:
    import numpy as np
    from slate_core import Basis, Ctype
    from slate_core.metadata import (
        AxisDirections,
        EvenlySpacedLengthMetadata,
        EvenlySpacedVolumeMetadata,
    )
    from slate_quantum.metadata import EigenvalueMetadata, RepeatedVolumeMetadata
    from slate_quantum.noise import (
        DiagonalNoiseOperatorList,
        NoiseOperatorList,
    )
    from slate_quantum.operator import Operator, OperatorBasis, Potential
    from slate_quantum.state import StateWithMetadata

    from adsorbate_simulation.system._config import SimulationConfig
    from adsorbate_simulation.system._system import System


@dataclass(frozen=True)
class SimulationCondition[
    S: System[Any] = System[Any],
    C: SimulationConfig = SimulationConfig,
]:
    system: S
    config: C

    def with_system[S_: System[Any]](self, system: S_) -> SimulationCondition[S_, C]:
        """Create a new condition with different system."""
        return SimulationCondition(system, self.config)

    def with_config[C_: SimulationConfig](
        self, config: C_
    ) -> SimulationCondition[S, C_]:
        """Create a new condition with different config."""
        return SimulationCondition(self.system, config)

    def with_temperature(self, temperature: float) -> SimulationCondition[S, C]:
        """Create a new condition with different temperature."""
        return self.with_config(self.config.with_temperature(temperature))

    @property
    def potential(
        self,
    ) -> Potential[
        EvenlySpacedLengthMetadata,
        AxisDirections,
        Ctype[Never],
        np.dtype[np.complexfloating],
    ]:
        """Get the potential for the simulation."""
        return self.system.get_potential(self.config.simulation_basis)

    @property
    def hamiltonian(
        self,
    ) -> Operator[
        OperatorBasis[EvenlySpacedVolumeMetadata], np.dtype[np.complexfloating]
    ]:
        system_hamiltonian = self.system.get_hamiltonian(self.config.simulation_basis)
        shift = self.config.get_hamiltonian_shift(system_hamiltonian)
        return system_hamiltonian + shift

    @property
    def fundamental_metadata(self) -> RepeatedVolumeMetadata:
        return self.config.simulation_basis.get_fundamental_metadata(self.system.cell)

    @property
    def initial_state(self) -> StateWithMetadata[EvenlySpacedVolumeMetadata]:
        return self.config.get_initial_state(self.system)

    @overload
    def get_environment_operators[C_: IsotropicSimulationConfig](
        self: SimulationCondition[Any, C_],
    ) -> DiagonalNoiseOperatorList[EigenvalueMetadata, EvenlySpacedVolumeMetadata]: ...
    @overload
    def get_environment_operators(
        self,
    ) -> NoiseOperatorList[EigenvalueMetadata, EvenlySpacedVolumeMetadata]: ...

    def get_environment_operators(
        self,
    ) -> NoiseOperatorList[EigenvalueMetadata, EvenlySpacedVolumeMetadata]:
        metadata = self.fundamental_metadata
        return self.config.environment.get_operators(metadata)

    @property
    def temperature_corrected_operators(
        self,
    ) -> NoiseOperatorList[EigenvalueMetadata, EvenlySpacedVolumeMetadata]:
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

    @property
    def eta(self) -> float:
        """The friction coefficient of the system."""
        return self.config.environment.eta

    @property
    def gamma(self) -> float:
        """The friction coefficient of the system."""
        return self.config.environment.gamma(self.mass)

    @property
    def simulation_basis(self) -> Basis[EvenlySpacedVolumeMetadata]:
        return self.config.simulation_basis.get_simulation_basis(self.system.cell)

    @property
    def operator_basis(
        self,
    ) -> OperatorBasis[EvenlySpacedVolumeMetadata]:
        """The simulation basis of the simulation."""
        return self.config.simulation_basis.get_operator_basis(self.system.cell)

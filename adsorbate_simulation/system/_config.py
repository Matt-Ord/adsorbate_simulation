from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, override

import numpy as np
from scipy.constants import hbar  # type: ignore lib
from slate import basis as _basis
from slate import metadata, tuple_basis
from slate.basis import diagonal_basis
from slate_quantum import state
from slate_quantum.metadata import RepeatedVolumeMetadata, eigenvalue_basis
from slate_quantum.noise import (
    DiagonalNoiseOperatorList,
    IsotropicNoiseKernel,
    NoiseKernel,
    NoiseOperatorList,
    build,
)
from slate_quantum.operator import OperatorList

from adsorbate_simulation.system._potential import (
    HarmonicPotential,
    SimulationPotential,
)
from adsorbate_simulation.util._eta import eta_from_gamma

if TYPE_CHECKING:
    from slate.metadata import SpacedVolumeMetadata
    from slate_quantum import Operator, State

    from adsorbate_simulation.system._basis import SimulationBasis, SimulationCell
    from adsorbate_simulation.system._system import System


class Environment(ABC):
    """Represents the environment of the system."""

    @property
    @abstractmethod
    def eta(self) -> float: ...
    @abstractmethod
    def get_operators(
        self, metadata: SpacedVolumeMetadata
    ) -> NoiseOperatorList[SpacedVolumeMetadata]: ...

    def get_temperature_corrected_operators(
        self,
        hamiltonian: Operator[SpacedVolumeMetadata, np.complexfloating],
        temperature: float,
    ) -> NoiseOperatorList[SpacedVolumeMetadata]:
        metadata = (hamiltonian.basis.metadata())[0]
        operators = build.temperature_corrected_operators(
            hamiltonian, self.get_operators(metadata), temperature, self.eta
        )
        return operators.with_operator_basis(_basis.as_tuple_basis(operators.basis)[1])

    def get_hamiltonian_shift(
        self, hamiltonian: Operator[SpacedVolumeMetadata, np.complexfloating]
    ) -> Operator[SpacedVolumeMetadata, np.complexfloating]:
        return build.hamiltonian_shift(
            hamiltonian, self.get_operators(hamiltonian.basis.metadata()[0]), self.eta
        )

    def get_noise_kernel(
        self, metadata: SpacedVolumeMetadata
    ) -> NoiseKernel[SpacedVolumeMetadata, np.complexfloating]:
        return NoiseKernel.from_operators(self.get_operators(metadata))

    @override
    def __eq__(self, value: object) -> bool:
        return isinstance(value, Environment)

    @override
    def __hash__(self) -> int:
        return 0


class IsotropicEnvironment(Environment):
    @override
    @abstractmethod
    def get_operators(
        self, metadata: SpacedVolumeMetadata
    ) -> DiagonalNoiseOperatorList[SpacedVolumeMetadata]: ...
    @override
    def get_noise_kernel(
        self, metadata: SpacedVolumeMetadata
    ) -> IsotropicNoiseKernel[SpacedVolumeMetadata, np.complexfloating]:
        return IsotropicNoiseKernel.from_operators(self.get_operators(metadata))


class ClosedEnvironment(IsotropicEnvironment):
    @property
    @override
    def eta(self) -> float:
        return 0

    @override
    def get_operators(
        self, metadata: SpacedVolumeMetadata
    ) -> DiagonalNoiseOperatorList[SpacedVolumeMetadata]:
        basis = _basis.from_metadata(metadata)
        return OperatorList(
            tuple_basis(
                (
                    eigenvalue_basis(np.array([])),
                    diagonal_basis((basis, basis.dual_basis())),
                )
            ),
            np.array([]),
        )

    @override
    def __eq__(self, value: object) -> bool:
        return isinstance(value, ClosedEnvironment)

    @override
    def __hash__(self) -> int:
        return 0


@dataclass(frozen=True, kw_only=True)
class CaldeiraLeggettEnvironment(IsotropicEnvironment):
    _eta: float = field(default=0, kw_only=True)

    @property
    @override
    def eta(self) -> float:
        return self._eta

    def with_eta(self, eta: float) -> Self:
        r"""Create a new environment with different friction \eta."""
        return type(self)(_eta=eta)

    @classmethod
    def from_gamma(cls, gamma: float, mass: float) -> Self:
        r"""Create a new environment with different damping \gamma."""
        return cls(_eta=eta_from_gamma(gamma, mass))

    @override
    def get_operators(
        self, metadata: SpacedVolumeMetadata
    ) -> DiagonalNoiseOperatorList[SpacedVolumeMetadata]:
        operators = build.caldeira_leggett_operators(metadata)
        return operators.with_operator_basis(operators.basis[1].inner)


@dataclass(frozen=True, kw_only=True)
class PeriodicCaldeiraLeggettEnvironment(IsotropicEnvironment):
    _eta: float = field(default=0, kw_only=True)

    @property
    @override
    def eta(self) -> float:
        return self._eta

    def with_eta(self, eta: float) -> Self:
        r"""Create a new environment with different friction \eta."""
        return type(self)(_eta=eta)

    @classmethod
    def from_gamma(cls, gamma: float, mass: float) -> Self:
        r"""Create a new environment with different damping \gamma."""
        return cls(_eta=eta_from_gamma(gamma, mass))

    @override
    def get_operators(
        self, metadata: SpacedVolumeMetadata
    ) -> DiagonalNoiseOperatorList[SpacedVolumeMetadata]:
        operators = build.real_periodic_caldeira_leggett_operators(metadata)
        return operators.with_operator_basis(operators.basis[1].inner)


class InitialState(ABC):
    """Configure the initial state of the system."""

    @abstractmethod
    def get_state(
        self, system: System[Any], basis: SimulationBasis
    ) -> State[SpacedVolumeMetadata]: ...


@dataclass(frozen=True, kw_only=True)
class CoherentInitialState(InitialState):
    """A Gaussian initial state, centered at the minimum of the potential."""

    width_factor: tuple[int, ...] | int = 6

    @override
    def get_state[P: SimulationPotential](
        self, system: System[P], basis: SimulationBasis
    ) -> State[SpacedVolumeMetadata]:
        widths = (
            (self.width_factor for _ in system.cell.lengths)
            if isinstance(self.width_factor, int)
            else self.width_factor
        )
        sigma_0 = tuple(
            ln / w for ln, w in zip(system.cell.lengths, widths, strict=True)
        )
        potential = system.potential.get_potential(system.cell, basis).as_diagonal()
        x_points = metadata.volume.fundamental_stacked_x_points(
            potential.basis.metadata()
        )
        min_point = np.argmin(potential.as_array())
        x_0 = tuple(x[min_point] for x in x_points)
        k_0 = tuple(0 for _ in system.cell.lengths)
        return state.build.coherent(potential.basis.metadata(), x_0, k_0, sigma_0)


@dataclass(frozen=True, kw_only=True)
class HarmonicCoherentInitialState(InitialState):
    """A Gaussian initial state, centered at the minimum of the potential."""

    @staticmethod
    def get_harmonic_width(frequency: float, mass: float) -> float:
        # we have 1/2 m omega **2 = 1/2 freq **2
        omega = frequency / np.sqrt(mass)
        return np.sqrt(hbar / (omega * mass))

    @override
    def get_state[P: SimulationPotential](
        self, system: System[P], basis: SimulationBasis
    ) -> State[SpacedVolumeMetadata]:
        assert isinstance(system.potential, HarmonicPotential)
        width = self.get_harmonic_width(system.potential.frequency, system.mass)
        sigma_0 = tuple(width for _ in system.cell.lengths)

        metadata = basis.get_fundamental_metadata(system.cell)
        x_0 = tuple(x / 2 for x in system.cell.lengths)
        k_0 = tuple(0 for _ in system.cell.lengths)
        return state.build.coherent(metadata, x_0, k_0, sigma_0)


@dataclass(frozen=True, kw_only=True)
class SimulationConfig:
    """Configure the simulation-specific detail of the system."""

    simulation_basis: SimulationBasis
    environment: Environment = field(default_factory=ClosedEnvironment, kw_only=True)
    temperature: float = field(default=150, kw_only=True)
    scattered_energy_range: tuple[float, float] = field(
        default=(-np.inf, np.inf),
        kw_only=True,
    )
    direction: tuple[int, ...] | None = field(default=None, kw_only=True)
    target_delta: float = field(default=1e-5, kw_only=True)
    initial_state: InitialState = field(
        default_factory=CoherentInitialState, kw_only=True
    )

    def __post_init__(self) -> None:
        assert self.scattered_energy_range[0] <= self.scattered_energy_range[1]

    def with_simulation_basis(self, simulation_basis: SimulationBasis) -> Self:
        """Create a new config with different simulation basis."""
        return type(self)(
            simulation_basis=simulation_basis,
            environment=self.environment,
            temperature=self.temperature,
            scattered_energy_range=self.scattered_energy_range,
            direction=self.direction,
            target_delta=self.target_delta,
            initial_state=self.initial_state,
        )

    def with_environment(self, environment: Environment) -> Self:
        """Create a new config with different environment."""
        return type(self)(
            simulation_basis=self.simulation_basis,
            environment=environment,
            temperature=self.temperature,
            scattered_energy_range=self.scattered_energy_range,
            direction=self.direction,
            target_delta=self.target_delta,
            initial_state=self.initial_state,
        )

    def with_temperature(self, temperature: float) -> Self:
        """Create a new config with different temperature."""
        return type(self)(
            simulation_basis=self.simulation_basis,
            environment=self.environment,
            temperature=temperature,
            scattered_energy_range=self.scattered_energy_range,
            direction=self.direction,
            target_delta=self.target_delta,
            initial_state=self.initial_state,
        )

    def with_scattered_energy_range(
        self,
        energy_range: tuple[float, float],
    ) -> Self:
        """Create a new config with different scattered energy range."""
        return type(self)(
            simulation_basis=self.simulation_basis,
            environment=self.environment,
            temperature=self.temperature,
            scattered_energy_range=energy_range,
            direction=self.direction,
            target_delta=self.target_delta,
            initial_state=self.initial_state,
        )

    def with_direction(self, direction: tuple[int, ...]) -> Self:
        """Create a new config with different direction."""
        return type(self)(
            simulation_basis=self.simulation_basis,
            environment=self.environment,
            temperature=self.temperature,
            scattered_energy_range=self.scattered_energy_range,
            direction=direction,
            target_delta=self.target_delta,
            initial_state=self.initial_state,
        )

    def with_target_delta(self, target_delta: float) -> Self:
        """Create a new config with different target delta."""
        return type(self)(
            simulation_basis=self.simulation_basis,
            environment=self.environment,
            temperature=self.temperature,
            scattered_energy_range=self.scattered_energy_range,
            direction=self.direction,
            target_delta=target_delta,
            initial_state=self.initial_state,
        )

    def with_initial_state(self, initial_state: InitialState) -> Self:
        """Create a new config with different initial state."""
        return type(self)(
            simulation_basis=self.simulation_basis,
            environment=self.environment,
            temperature=self.temperature,
            scattered_energy_range=self.scattered_energy_range,
            direction=self.direction,
            target_delta=self.target_delta,
            initial_state=initial_state,
        )

    def get_fundamental_metadata(self, cell: SimulationCell) -> RepeatedVolumeMetadata:
        return self.simulation_basis.get_fundamental_metadata(cell)

    def get_temperature_corrected_operators(
        self, hamiltonian: Operator[SpacedVolumeMetadata, np.complexfloating]
    ) -> NoiseOperatorList[SpacedVolumeMetadata]:
        return self.environment.get_temperature_corrected_operators(
            hamiltonian, self.temperature
        )

    def get_hamiltonian_shift(
        self, hamiltonian: Operator[SpacedVolumeMetadata, np.complexfloating]
    ) -> Operator[SpacedVolumeMetadata, np.complexfloating]:
        return self.environment.get_hamiltonian_shift(hamiltonian)

    def get_initial_state(self, system: System[Any]) -> State[SpacedVolumeMetadata]:
        return self.initial_state.get_state(system, self.simulation_basis)


@dataclass(frozen=True, kw_only=True)
class IsotropicSimulationConfig(SimulationConfig):
    environment: IsotropicEnvironment = field(default_factory=ClosedEnvironment)

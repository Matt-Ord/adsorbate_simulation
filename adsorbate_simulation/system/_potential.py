from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Never, override

import numpy as np
from slate_quantum import operator

if TYPE_CHECKING:
    from slate_core import Ctype
    from slate_core.metadata import AxisDirections, SpacedLengthMetadata
    from slate_quantum.operator import Potential

    from adsorbate_simulation.system._basis import SimulationBasis, SimulationCell


@dataclass(frozen=True, kw_only=True)
class SimulationPotential(ABC):
    """Represents a potential used in a simulation."""

    @abstractmethod
    def get_repeat_potential(
        self,
        cell: SimulationCell,
        simulation_basis: SimulationBasis,
    ) -> Potential[
        SpacedLengthMetadata,
        AxisDirections,
        Ctype[Never],
        np.dtype[np.complexfloating],
    ]:
        """Get the potential for the repeat cell of the simulation."""

    def get_potential(
        self,
        cell: SimulationCell,
        simulation_basis: SimulationBasis,
    ) -> Potential[
        SpacedLengthMetadata,
        AxisDirections,
        Ctype[Never],
        np.dtype[np.complexfloating],
    ]:
        """Get the potential for the simulation."""
        potential = self.get_repeat_potential(cell, simulation_basis)
        return operator.repeat_potential(potential, simulation_basis.shape)


@dataclass(frozen=True, kw_only=True)
class CosPotential(SimulationPotential):
    """A simple cosine potential, with a single barrier height."""

    barrier_height: float

    def with_barrier_height(self, barrier_height: float) -> CosPotential:  # noqa: PLR6301
        """Create a new system with different barrier height."""
        return CosPotential(
            barrier_height=barrier_height,
        )

    @override
    def get_repeat_potential(
        self,
        cell: SimulationCell,
        simulation_basis: SimulationBasis,
    ) -> Potential[
        SpacedLengthMetadata,
        AxisDirections,
        Ctype[Never],
        np.dtype[np.complexfloating],
    ]:
        return operator.build.cos_potential(
            simulation_basis.get_repeat_metadata(cell),
            self.barrier_height,
        )


@dataclass(frozen=True, kw_only=True)
class FCCPotential(SimulationPotential):
    """A simple cosine potential, with a single barrier height."""

    top_site_energy: float

    def with_top_site_energy(self, top_site_energy: float) -> FCCPotential:  # noqa: PLR6301
        """Create a new system with different barrier height."""
        return FCCPotential(
            top_site_energy=top_site_energy,
        )

    def with_bridge_site_energy(self, bridge_site_energy: float) -> FCCPotential:  # noqa: PLR6301
        """Create a new system with different barrier height."""
        return FCCPotential(
            top_site_energy=9 * bridge_site_energy,
        )

    @override
    def get_repeat_potential(
        self,
        cell: SimulationCell,
        simulation_basis: SimulationBasis,
    ) -> Potential[
        SpacedLengthMetadata,
        AxisDirections,
        Ctype[Never],
        np.dtype[np.complexfloating],
    ]:
        return operator.build.fcc_potential(
            simulation_basis.get_repeat_metadata(cell),
            self.top_site_energy,
        )


class FreePotential(CosPotential):
    """A potential with no barrier height."""

    def __init__(self) -> None:
        super().__init__(barrier_height=0.0)


@dataclass(frozen=True, kw_only=True)
class HarmonicPotential(SimulationPotential):
    """A potential with a harmonic potential shape."""

    frequency: float

    def with_frequency(self, frequency: float) -> HarmonicPotential:  # noqa: PLR6301
        """Create a new system with different frequency."""
        return HarmonicPotential(frequency=frequency)

    def _potential_fn(
        self,
        lengths: tuple[float, ...],
        points: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
    ) -> np.ndarray[Any, np.dtype[np.complexfloating]]:
        points = tuple((p % d) - d / 2 for p, d in zip(points, lengths, strict=True))
        return (0.5 * self.frequency**2 * np.linalg.norm(points, axis=0) ** 2).astype(
            np.complexfloating
        )

    @override
    def get_repeat_potential(
        self,
        cell: SimulationCell,
        simulation_basis: SimulationBasis,
    ) -> Potential[
        SpacedLengthMetadata, AxisDirections, Ctype[Never], np.dtype[np.complexfloating]
    ]:
        metadata = simulation_basis.get_repeat_metadata(cell)
        return operator.build.potential_from_function(
            metadata,
            lambda x: self._potential_fn(cell.lengths, x),
        )

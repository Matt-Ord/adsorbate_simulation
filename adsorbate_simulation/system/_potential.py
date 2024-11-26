from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, override

import numpy as np
from scipy.constants import electron_volt  # type: ignore  no type hints for scipy
from slate.metadata import AxisDirections, SpacedVolumeMetadata
from slate_quantum.model.operator import build_cos_potential, repeat_potential

if TYPE_CHECKING:
    from slate_quantum.model import Potential

    from adsorbate_simulation.system._basis import SimulationBasis


@dataclass(frozen=True, kw_only=True)
class SimulationPotential(ABC):
    """Represents a potential used in a simulation."""

    lengths: tuple[float, ...]
    directions: AxisDirections

    def with_lengths(self: Self, lengths: tuple[float, ...]) -> SimulationPotential:
        """Create a new system with different lengths."""
        return type(self)(lengths=lengths, directions=self.directions)

    def with_directions(self: Self, directions: AxisDirections) -> SimulationPotential:
        """Create a new system with different directions."""
        return type(self)(
            lengths=self.lengths,
            directions=directions,
        )

    @abstractmethod
    def get_repeat_potential(
        self: Self,
        simulation_basis: SimulationBasis,
    ) -> Potential[SpacedVolumeMetadata, np.complex128]:
        """Get the potential for the repeat cell of the simulation."""

    def get_potential(
        self: Self,
        simulation_basis: SimulationBasis,
    ) -> Potential[SpacedVolumeMetadata, np.complex128]:
        """Get the potential for the simulation."""
        potential = self.get_repeat_potential(simulation_basis)
        return repeat_potential(potential, simulation_basis.shape)


@dataclass(frozen=True, kw_only=True)
class CosPotential(SimulationPotential):
    """A simple cosine potential, with a single barrier height."""

    barrier_height: float

    def with_barrier_height(self: Self, barrier_height: float) -> CosPotential:
        """Create a new system with different barrier height."""
        return CosPotential(
            lengths=self.lengths,
            directions=self.directions,
            barrier_height=barrier_height,
        )

    @override
    def with_lengths(self: Self, lengths: tuple[float, ...]) -> CosPotential:
        return CosPotential(
            lengths=lengths,
            directions=self.directions,
            barrier_height=self.barrier_height,
        )

    @override
    def with_directions(self: Self, directions: AxisDirections) -> CosPotential:
        return CosPotential(
            lengths=self.lengths,
            directions=directions,
            barrier_height=self.barrier_height,
        )

    @override
    def get_repeat_potential(
        self: Self,
        simulation_basis: SimulationBasis,
    ) -> Potential[SpacedVolumeMetadata, np.complex128]:
        return build_cos_potential(
            simulation_basis.get_repeat_basis(self.lengths, self.directions).metadata(),
            self.barrier_height,
        )


class FreePotential(CosPotential):
    """A potential with no barrier height."""

    def __init__(self, lengths: tuple[float, ...], directions: AxisDirections) -> None:
        super().__init__(lengths=lengths, directions=directions, barrier_height=0.0)


LI_CU_COS_POTENTIAL = CosPotential(
    lengths=((1 / np.sqrt(3)) * 3.615e-10,),
    directions=AxisDirections(vectors=(np.array([1]),)),
    barrier_height=45e-3 * electron_volt,
)

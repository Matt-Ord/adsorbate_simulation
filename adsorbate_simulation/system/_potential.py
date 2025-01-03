from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from scipy.constants import electron_volt  # type: ignore  no type hints for scipy
from slate_quantum import operator

if TYPE_CHECKING:
    import numpy as np
    from slate.metadata import AxisDirections, SpacedLengthMetadata
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
    ) -> Potential[SpacedLengthMetadata, AxisDirections, np.complexfloating]:
        """Get the potential for the repeat cell of the simulation."""

    def get_potential(
        self,
        cell: SimulationCell,
        simulation_basis: SimulationBasis,
    ) -> Potential[SpacedLengthMetadata, AxisDirections, np.complexfloating]:
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
    ) -> Potential[SpacedLengthMetadata, AxisDirections, np.complexfloating]:
        return operator.build.cos_potential(
            simulation_basis.get_repeat_basis(cell).metadata(),
            self.barrier_height,
        )


class FreePotential(CosPotential):
    """A potential with no barrier height."""

    def __init__(self) -> None:
        super().__init__(barrier_height=0.0)


LI_CU_COS_POTENTIAL = CosPotential(
    barrier_height=45e-3 * electron_volt,
)

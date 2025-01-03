from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import numpy as np
from scipy.constants import hbar  # type: ignore  no type hints for scipy
from slate.metadata import AxisDirections, SpacedLengthMetadata, SpacedVolumeMetadata
from slate_quantum import operator

from adsorbate_simulation.system._basis import SimulationCell
from adsorbate_simulation.system._potential import (
    LI_CU_COS_POTENTIAL,
    FreePotential,
    SimulationPotential,
)

if TYPE_CHECKING:
    from slate_quantum.operator import Operator, Potential

    from ._basis import SimulationBasis


@dataclass(frozen=True)
class System[P: SimulationPotential]:
    """Represents the properties of a Periodic System."""

    id: str
    """A unique ID, for use in caching"""
    mass: float
    potential: P
    cell: SimulationCell

    def with_mass(self, mass: float) -> System[P]:
        """Create a new system with different mass."""
        return System(self.id, mass, self.potential, self.cell)

    def with_potential[P1: SimulationPotential](self, potential: P1) -> System[P1]:
        """Create a new system with different potential."""
        return System(self.id, self.mass, potential, self.cell)

    def with_cell(self, cell: SimulationCell) -> System[P]:
        """Create a new system with different cell."""
        return System(self.id, self.mass, self.potential, cell)

    @override
    def __hash__(self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(self.id.encode())
        id_hash = int.from_bytes(h.digest(), "big")

        return hash((id_hash, self.mass, self.potential))

    def get_potential(
        self,
        simulation_basis: SimulationBasis,
    ) -> Potential[SpacedLengthMetadata, AxisDirections, np.complexfloating]:
        """Get the potential for the simulation."""
        return self.potential.get_potential(self.cell, simulation_basis)

    def get_hamiltonian(
        self,
        simulation_basis: SimulationBasis,
    ) -> Operator[SpacedVolumeMetadata, np.complexfloating]:
        """Get the hamiltonian for the simulation."""
        return operator.build_kinetic_hamiltonian(
            self.get_potential(simulation_basis), self.mass
        )


LI_CU_UNIT_CELL = SimulationCell(
    lengths=((1 / np.sqrt(3)) * 3.615e-10,),
    directions=AxisDirections(vectors=(np.array([1]),)),
)
LI_CU_SYSTEM_1D = System("LI_CU", 1, LI_CU_COS_POTENTIAL, LI_CU_UNIT_CELL)

DIMENSIONLESS_UNIT_CELL = SimulationCell(
    lengths=(2 * np.pi,),
    directions=AxisDirections(vectors=(np.array([1]),)),
)
DIMENSIONLESS_SYSTEM_1D = System(
    "DIMENSIONLESS",
    hbar**2,
    FreePotential(),
    DIMENSIONLESS_UNIT_CELL,
)

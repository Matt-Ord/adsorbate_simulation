from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from slate_quantum import operator

from adsorbate_simulation.system._potential import (
    SimulationPotential,
)

if TYPE_CHECKING:
    import numpy as np
    from slate.metadata import (
        AxisDirections,
        SpacedLengthMetadata,
        SpacedVolumeMetadata,
    )
    from slate_quantum.operator import Operator, Potential

    from adsorbate_simulation.system._basis import SimulationCell

    from ._basis import SimulationBasis


@dataclass(frozen=True)
class System[P: SimulationPotential]:
    """Represents the properties of a Periodic System."""

    mass: float
    potential: P
    cell: SimulationCell

    def with_mass(self, mass: float) -> System[P]:
        """Create a new system with different mass."""
        return System(mass, self.potential, self.cell)

    def with_potential[P1: SimulationPotential](self, potential: P1) -> System[P1]:
        """Create a new system with different potential."""
        return System(self.mass, potential, self.cell)

    def with_cell(self, cell: SimulationCell) -> System[P]:
        """Create a new system with different cell."""
        return System(self.mass, self.potential, cell)

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

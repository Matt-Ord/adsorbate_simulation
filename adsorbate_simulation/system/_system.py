from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, override

import numpy as np
from scipy.constants import hbar  # type: ignore  no type hints for scipy
from slate.metadata import AxisDirections, Metadata2D, SpacedVolumeMetadata
from slate_quantum.model.operator import build_kinetic_hamiltonian

from adsorbate_simulation.system._potential import (
    LI_CU_COS_POTENTIAL,
    FreePotential,
    SimulationPotential,
)

if TYPE_CHECKING:
    from slate_quantum.model import Operator, Potential

    from ._basis import SimulationBasis


@dataclass(frozen=True)
class System[P: SimulationPotential]:
    """Represents the properties of a Periodic System."""

    id: str
    """A unique ID, for use in caching"""
    mass: float
    potential: P

    def with_mass(self: Self, mass: float) -> System[P]:
        """Create a new system with different mass."""
        return System(self.id, mass, self.potential)

    def with_potential[P1: SimulationPotential](
        self: Self, potential: P1
    ) -> System[P1]:
        """Create a new system with different potential."""
        return System(self.id, self.mass, potential)

    @override
    def __hash__(self: Self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(self.id.encode())
        id_hash = int.from_bytes(h.digest(), "big")

        return hash((id_hash, self.mass, self.potential))

    def get_potential(
        self: Self,
        simulation_basis: SimulationBasis,
    ) -> Potential[SpacedVolumeMetadata, np.complex128]:
        """Get the potential for the simulation."""
        return self.potential.get_potential(simulation_basis)

    def get_hamiltonian(
        self: Self,
        simulation_basis: SimulationBasis,
    ) -> Operator[
        Metadata2D[SpacedVolumeMetadata, SpacedVolumeMetadata, None], np.complex128
    ]:
        """Get the hamiltonian for the simulation."""
        return build_kinetic_hamiltonian(
            self.get_potential(simulation_basis), self.mass
        )


LI_CU_SYSTEM_1D = System(
    "LI_CU",
    1,
    LI_CU_COS_POTENTIAL,
)
DIMENSIONLESS_SYSTEM_1D = System(
    "DIMENSIONLESS",
    hbar**2,
    FreePotential((2 * np.pi,), AxisDirections(vectors=(np.array([1.0]),))),
)

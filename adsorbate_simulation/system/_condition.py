from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from slate.metadata import SpacedVolumeMetadata
    from slate_quantum.model import Potential

    from adsorbate_simulation.system._config import SimulationConfig
    from adsorbate_simulation.system._potential import SimulationPotential
    from adsorbate_simulation.system._system import System


@dataclass
class SimulationCondition[P: SimulationPotential]:
    system: System[P]
    config: SimulationConfig

    def with_system[P1: SimulationPotential](
        self, system: System[P1]
    ) -> SimulationCondition[P1]:
        """Create a new condition with different system."""
        return SimulationCondition(system, self.config)

    def with_config(self, config: SimulationConfig) -> SimulationCondition[P]:
        """Create a new condition with different config."""
        return SimulationCondition(self.system, config)

    def get_potential(
        self,
    ) -> Potential[SpacedVolumeMetadata, np.complex128]:
        """Get the potential for the simulation."""
        return self.system.get_potential(self.config.simulation_basis)

    @property
    def temperature(self) -> float:
        """The temperature of the simulation."""
        return self.config.temperature

    @property
    def direction(self) -> tuple[int, ...]:
        """The direction of the scattering of the simulation."""
        return self.config.direction

    @property
    def scattered_energy_range(self) -> tuple[float, float]:
        """The scattered energy range of the simulation."""
        return self.config.scattered_energy_range

    @property
    def mass(self) -> float:
        """The mass of the system."""
        return self.system.mass

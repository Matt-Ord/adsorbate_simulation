from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from slate import Basis, SimpleMetadata
from slate.metadata import Metadata2D
from slate.util import cached
from slate_quantum import StateList, state
from slate_quantum.dynamics import solve_stochastic_schrodinger_equation_banded
from slate_quantum.metadata import TimeMetadata

if TYPE_CHECKING:
    from slate.metadata import SpacedVolumeMetadata

    from adsorbate_simulation.system import (
        FreePotential,
        IsotropicSimulationConfig,
        SimulationCondition,
        System,
    )


def _get_simulation_path[M: TimeMetadata](
    condition: SimulationCondition[System[FreePotential], IsotropicSimulationConfig],
    times: Basis[M, np.complexfloating],
) -> Path:
    directory = Path(os.path.realpath(__file__)).parent
    return directory / "data" / f"{hash(condition)}.{hash(times)}.states"


@cached(_get_simulation_path)
def run_simulation[M: TimeMetadata](
    condition: SimulationCondition[System[FreePotential], IsotropicSimulationConfig],
    times: Basis[M, np.complexfloating],
) -> StateList[Metadata2D[SimpleMetadata, M, None], SpacedVolumeMetadata]:
    """Run a stochastic simulation."""
    hamiltonian = condition.hamiltonian
    hamiltonian = hamiltonian.with_basis(
        condition.config.simulation_basis.get_operator_basis(condition.system.cell)
    )

    environment_operators = condition.temperature_corrected_operators
    initial_state = state.build_coherent_state(
        hamiltonian.basis.metadata()[0], (0,), (0,), (np.pi,)
    )

    return solve_stochastic_schrodinger_equation_banded(
        initial_state,
        times,
        hamiltonian,
        environment_operators,
        method="Order2ExplicitWeak",
        target_delta=2e-5,
    )

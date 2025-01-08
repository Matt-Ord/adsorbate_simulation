from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from slate.util import cached
from slate_quantum import StateList, state
from slate_quantum.dynamics import solve_stochastic_schrodinger_equation_banded
from slate_quantum.metadata import (
    TimeMetadata,
)

from adsorbate_simulation.system._condition import SimulationCondition

if TYPE_CHECKING:
    from slate import Basis, SimpleMetadata
    from slate.metadata import Metadata2D, SpacedVolumeMetadata

    from adsorbate_simulation.system import (
        IsotropicSimulationConfig,
        SimulationCondition,
        System,
    )


def _get_simulation_path[M: TimeMetadata](
    condition: SimulationCondition[System[Any], IsotropicSimulationConfig],
    times: Basis[M, np.complexfloating],
) -> Path:
    directory = Path(os.path.realpath(sys.argv[0])).parent
    return directory / "data" / f"{hash(condition)}.{hash(times)}.states"


@cached(_get_simulation_path)
def run_stochastic_simulation[M: TimeMetadata](
    condition: SimulationCondition[System[Any], IsotropicSimulationConfig],
    times: Basis[M, np.complexfloating],
) -> StateList[Metadata2D[SimpleMetadata, M, None], SpacedVolumeMetadata]:
    """Run a stochastic simulation."""
    hamiltonian = condition.hamiltonian
    hamiltonian = hamiltonian.with_basis(
        condition.config.simulation_basis.get_operator_basis(condition.system.cell)
    )

    environment_operators = condition.temperature_corrected_operators
    # TODO: specify initial state strategy in config
    width = condition.system.cell.lengths[0] / 2
    initial_state = state.build_coherent_state(
        hamiltonian.basis.metadata()[0], (0,), (0,), (width,)
    )

    return solve_stochastic_schrodinger_equation_banded(
        initial_state,
        times,
        hamiltonian,
        environment_operators,
        method="Order2ExplicitWeak",
        target_delta=condition.config.target_delta,
    )

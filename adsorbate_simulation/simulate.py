from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slate_core.util import cached
from slate_quantum.dynamics import solve_stochastic_schrodinger_equation_banded
from slate_quantum.metadata import (
    TimeMetadata,
)

if TYPE_CHECKING:
    from slate_core import Basis, SimpleMetadata, TupleMetadata
    from slate_core.metadata import SpacedVolumeMetadata
    from slate_quantum import StateList

    from adsorbate_simulation.system import (
        IsotropicSimulationConfig,
        SimulationCondition,
        System,
    )


def _get_simulation_path[M: TimeMetadata](
    condition: SimulationCondition[System[Any], IsotropicSimulationConfig],
    times: Basis[M],
) -> Path:
    directory = Path(os.path.realpath(sys.argv[0])).parent
    return directory / "data" / f"{hash(condition)}.{hash(times)}.states"


@cached(_get_simulation_path)
def run_stochastic_simulation[M: TimeMetadata](
    condition: SimulationCondition[System[Any], IsotropicSimulationConfig],
    times: Basis[M],
) -> StateList[
    Basis[
        TupleMetadata[
            tuple[TupleMetadata[tuple[SimpleMetadata, M], None], SpacedVolumeMetadata]
        ]
    ]
]:
    """Run a stochastic simulation."""
    hamiltonian = condition.hamiltonian
    hamiltonian = hamiltonian.with_basis(condition.operator_basis)
    environment_operators = condition.temperature_corrected_operators
    initial_state = condition.get_initial_state()

    return solve_stochastic_schrodinger_equation_banded(
        initial_state,
        times,
        hamiltonian,
        environment_operators,
        method="Order2ExplicitWeak",
        target_delta=condition.config.target_delta,
    )

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slate_core import basis
from slate_core.util import cached
from slate_quantum.dynamics import (
    simulate_caldeira_leggett_realizations,
    solve_stochastic_schrodinger_equation_banded,
)
from slate_quantum.dynamics.caldeira_leggett import CaldeiraLeggettCondition
from slate_quantum.metadata import (
    TimeMetadata,
)

if TYPE_CHECKING:
    from slate_core import Basis, SimpleMetadata, TupleMetadata
    from slate_core.metadata import EvenlySpacedVolumeMetadata
    from slate_quantum import StateList

    from adsorbate_simulation.system import (
        IsotropicSimulationConfig,
        SimulationCondition,
        System,
    )
    from adsorbate_simulation.system._config import CaldeiraLeggettSimulationConfig


def _get_simulation_path[M: TimeMetadata, C: IsotropicSimulationConfig](
    condition: SimulationCondition[System[Any], C],
    times: Basis[M],
) -> Path:
    directory = Path(os.path.realpath(sys.argv[0])).parent
    return directory / "data" / f"{hash(condition)}.{hash(times)}.states"


@cached(_get_simulation_path)
def run_stochastic_simulation[M: TimeMetadata, C: IsotropicSimulationConfig](
    condition: SimulationCondition[System[Any], C],
    times: Basis[M],
) -> StateList[
    Basis[
        TupleMetadata[
            tuple[
                TupleMetadata[tuple[SimpleMetadata, M], None],
                EvenlySpacedVolumeMetadata,
            ]
        ]
    ]
]:
    """Run a stochastic simulation."""
    return solve_stochastic_schrodinger_equation_banded(
        condition.initial_state,
        times,
        condition.hamiltonian.with_basis(condition.operator_basis),
        condition.temperature_corrected_operators,
        method="Order2ExplicitWeak",
        target_delta=condition.config.target_delta,
    )


def _get_simulation_path_cl[M: TimeMetadata, C: CaldeiraLeggettSimulationConfig](
    condition: SimulationCondition[System[Any], C],
    times: Basis[M],
) -> Path:
    directory = Path(os.path.realpath(sys.argv[0])).parent
    return directory / "data" / f"{hash(condition)}.{hash(times)}.states"


@cached(_get_simulation_path_cl)
def run_caldeira_leggett_simulation[
    M: TimeMetadata,
    C: CaldeiraLeggettSimulationConfig,
](
    condition: SimulationCondition[System[Any], C],
    times: Basis[M],
) -> StateList[
    Basis[
        TupleMetadata[
            tuple[
                TupleMetadata[tuple[SimpleMetadata, M], None],
                EvenlySpacedVolumeMetadata,
            ]
        ]
    ]
]:
    """Run a Caldeira-Leggett simulation."""
    cl_condition = CaldeiraLeggettCondition(
        mass=condition.mass,
        temperature=condition.temperature,
        friction=condition.gamma,
        initial_state=condition.initial_state,
        potential=condition.potential,
    )
    out = simulate_caldeira_leggett_realizations(cl_condition, times, n_realizations=1)
    return out.with_basis(basis.from_metadata(out.basis.metadata()).upcast())

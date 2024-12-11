from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore libary
from slate.plot import animate_data_over_list_1d_x
from slate_quantum import state
from slate_quantum.dynamics import (
    solve_schrodinger_equation_decomposition,
    solve_stochastic_schrodinger_equation_banded,
)

from adsorbate_simulation.system import (
    DIMENSIONLESS_SYSTEM_1D,
    ClosedEnvironment,
    IsotropicSimulationConfig,
    SimulationBasis,
    SimulationCondition,
)
from adsorbate_simulation.util import spaced_time_basis

if __name__ == "__main__":
    # This is a simple example of simulating the system using the stochastic schrodinger
    # equation.
    # First we create a simulation condition for a free system in 1D.
    condition = SimulationCondition(
        DIMENSIONLESS_SYSTEM_1D,
        IsotropicSimulationConfig(
            simulation_basis=SimulationBasis((3,), (25,)),
            environment=ClosedEnvironment(),
            temperature=10 / Boltzmann,
        ),
    )

    hamiltonian = condition.hamiltonian
    times = spaced_time_basis(n=100, step=1000, dt=0.2 * np.pi * hbar)
    # We start the system in an gaussian state, centered at the origin.
    initial_state = state.build_coherent_state(
        condition.fundamental_metadata, (0,), (0,), (np.pi / 4,)
    )

    # We use the Closed environment.
    # This has no noise operators - the behavior should be
    # equivalent to the standard Schrodinger equation.
    environment_operators = condition.get_environment_operators()

    states = solve_stochastic_schrodinger_equation_banded(
        initial_state, times, hamiltonian, environment_operators
    )
    fig, ax, _anim0 = animate_data_over_list_1d_x(states, measure="real")
    states = solve_schrodinger_equation_decomposition(initial_state, times, hamiltonian)
    fig, ax, _anim1 = animate_data_over_list_1d_x(states, measure="real", ax=ax)
    for artist in _anim1.frame_seq:
        artist[0].set_color("C1")
        artist[0].set_linestyle("--")

    ax.set_title("Comparison of Stochastic and Schrodinger Evolution")
    fig.show()

    input()

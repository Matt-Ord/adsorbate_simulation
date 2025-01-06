from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore libary
from slate.plot import animate_data_over_list_1d_x
from slate_quantum import dynamics, state
from slate_quantum.dynamics import (
    solve_schrodinger_equation_decomposition,
)

from adsorbate_simulation.system import (
    DIMENSIONLESS_SYSTEM_1D,
    ClosedEnvironment,
    FundamentalSimulationBasis,
    IsotropicSimulationConfig,
    SimulationCondition,
)
from adsorbate_simulation.util import run_simulation, spaced_time_basis

if __name__ == "__main__":
    # This is a simple example of simulating the system using the stochastic schrodinger
    # equation.
    # First we create a simulation condition for a free system in 1D.
    condition = SimulationCondition(
        DIMENSIONLESS_SYSTEM_1D,
        IsotropicSimulationConfig(
            simulation_basis=FundamentalSimulationBasis(shape=(3,), resolution=(25,)),
            environment=ClosedEnvironment(),
            temperature=10 / Boltzmann,
        ),
    )

    # We use the Closed environment.
    # This has no noise - the behavior should be
    # equivalent to the standard Schrodinger equation.
    times = spaced_time_basis(n=100, dt=0.2 * np.pi * hbar)
    states = run_simulation(condition, times)
    states = dynamics.select_realization(states)
    fig, ax, _anim0 = animate_data_over_list_1d_x(states, measure="real")

    # We start the system in an gaussian state, centered at the origin.
    initial_state = state.build_coherent_state(
        condition.fundamental_metadata, (0,), (0,), (np.pi / 4,)
    )
    states = solve_schrodinger_equation_decomposition(
        initial_state, times, condition.hamiltonian
    )
    fig, ax, _anim1 = animate_data_over_list_1d_x(states, measure="real", ax=ax)
    for artist in _anim1.frame_seq:
        artist[0].set_color("C1")
        artist[0].set_linestyle("--")

    ax.set_title("Comparison of Stochastic and Schrodinger Evolution")
    fig.show()

    input()

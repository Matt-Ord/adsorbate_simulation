from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore libary
from slate import basis
from slate.plot import (
    animate_data_over_list_1d_k,
    animate_data_over_list_1d_x,
    plot_data_1d_x,
)
from slate_quantum import state
from slate_quantum.dynamics import (
    solve_stochastic_schrodinger_equation_banded,
)

from adsorbate_simulation.system import (
    DIMENSIONLESS_SYSTEM_1D,
    IsotropicSimulationConfig,
    PeriodicCaldeiraLeggettEnvironment,
    SimulationBasis,
    SimulationCondition,
)
from adsorbate_simulation.util import spaced_time_basis

if __name__ == "__main__":
    # This is a simple example of simulating the system using the stochastic schrodinger
    # equation.
    # First we create a simulation condition for a free system in 1D.
    # TODO: slightly unstable - but is this due to using too small number of states - can we stop this instability?  # noqa: FIX002
    condition = SimulationCondition(
        DIMENSIONLESS_SYSTEM_1D,
        IsotropicSimulationConfig(
            simulation_basis=SimulationBasis((4,), (25,)),
            environment=PeriodicCaldeiraLeggettEnvironment(_eta=4 / 3**2),
            temperature=10 / Boltzmann,
        ),
    )

    hamiltonian = condition.hamiltonian
    hamiltonian = hamiltonian.with_basis(
        basis.fundamental_transformed_tuple_basis_from_metadata(
            hamiltonian.basis.metadata(), is_dual=hamiltonian.basis.is_dual
        )
    )
    times = spaced_time_basis(n=100, step=1000, dt=0.2 * np.pi * hbar)
    # We start the system in an gaussian state, centered at the origin.
    initial_state = state.build_coherent_state(
        hamiltonian.basis.metadata()[0], (0,), (0,), (np.pi / 4,)
    )
    fig, ax, _ = plot_data_1d_x(initial_state, measure="abs")
    ax.set_title("Initial State - A Gaussian Wavepacket Centered at the Origin")
    fig.show()

    # We simulate the system using the stochastic Schrodinger equation.
    # We find a localized stochastic evolution of the wavepacket.
    environment_operators = condition.temperature_corrected_operators
    states = solve_stochastic_schrodinger_equation_banded(
        initial_state, times, hamiltonian, environment_operators
    )
    fig, ax, _anim0 = animate_data_over_list_1d_x(states, measure="abs")
    fig.show()
    fig, ax, _anim1 = animate_data_over_list_1d_k(states, measure="abs")
    fig.show()

    input()

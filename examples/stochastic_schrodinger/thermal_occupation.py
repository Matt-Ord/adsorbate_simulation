from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore unknown
from slate import array, basis, linalg
from slate.plot import get_figure, plot_array
from slate_quantum import state
from slate_quantum.dynamics import solve_stochastic_schrodinger_equation_banded

from adsorbate_simulation.system import (
    DIMENSIONLESS_SYSTEM_1D,
    IsotropicSimulationConfig,
    PeriodicCaldeiraLeggettEnvironment,
    SimulationBasis,
    SimulationCondition,
)
from adsorbate_simulation.util import (
    get_eigenvalue_occupation_hermitian,
    spaced_time_basis,
)

if __name__ == "__main__":
    # When we perform a simulation, we need to make sure we recover the
    # correct thermal occupation of the states.
    condition = SimulationCondition(
        DIMENSIONLESS_SYSTEM_1D,
        IsotropicSimulationConfig(
            simulation_basis=SimulationBasis((4,), (45,)),
            environment=PeriodicCaldeiraLeggettEnvironment(_eta=4 / 3**2),
            temperature=10 / Boltzmann,
        ),
    )

    # For a system in thermal equilibrium, the probability of a state
    # being occupied is given by the Boltzmann distribution.
    # We need to make sure to include enough states such that the
    # occupation of the highest energy state is negligible.
    # By plotting the occupation of the states, we can verify that
    # 25 is enough resolution for this system.
    #
    # Note that the 'shape'
    # of the basis is not important, only the resolution.
    hamiltonian = condition.hamiltonian
    hamiltonian = hamiltonian.with_basis(
        basis.fundamental_transformed_tuple_basis_from_metadata(
            hamiltonian.basis.metadata(), is_dual=hamiltonian.basis.is_dual
        )
    )
    times = spaced_time_basis(n=10000, step=1000, dt=0.2 * np.pi * hbar)
    diagonal_hamiltonian = linalg.into_diagonal_hermitian(hamiltonian)
    occupation = get_eigenvalue_occupation_hermitian(
        diagonal_hamiltonian, condition.config.temperature
    )
    fig, ax, line = plot_array(occupation)
    ax.set_title("Thermal occupation of the states")
    ax.set_xlabel("Energy /J")
    ax.set_ylabel("Occupation Probability")
    line.set_marker("x")
    fig.show()

    # Now we can test the thermal occupation of the states in our
    # periodic environment.
    initial_state = state.build_coherent_state(
        hamiltonian.basis.metadata()[0], (0,), (0,), (np.pi,)
    )
    environment_operators = condition.temperature_corrected_operators
    states = solve_stochastic_schrodinger_equation_banded(
        initial_state, times, hamiltonian, environment_operators
    )
    states = states.with_state_basis(diagonal_hamiltonian.basis.inner[0])
    average_occupation = array.cast_basis(
        state.get_average_occupations(states), occupation.basis
    )
    # We see that the true occupation of the states matches the
    # expected thermal occupation.
    fig, ax = get_figure()
    _, _, line = plot_array(occupation, ax=ax)
    _, _, line = plot_array(average_occupation, ax=ax)
    ax.set_title("True occupation of the states")
    ax.set_xlabel("Energy /J")
    ax.set_ylabel("Occupation Probability")
    line.set_marker("x")
    fig.show()
    input()

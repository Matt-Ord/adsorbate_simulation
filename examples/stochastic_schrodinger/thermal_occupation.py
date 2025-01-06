from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore unknown
from slate import array, linalg, plot
from slate_quantum import state

from adsorbate_simulation.fit import (
    TemperatureFitInfo,
    TemperatureFitMethod,
)
from adsorbate_simulation.system import (
    DIMENSIONLESS_SYSTEM_1D,
    IsotropicSimulationConfig,
    MomentumSimulationBasis,
    PeriodicCaldeiraLeggettEnvironment,
    SimulationCondition,
)
from adsorbate_simulation.util import (
    get_eigenvalue_occupation_hermitian,
    spaced_time_basis,
)
from examples.stochastic_schrodinger.util import run_simulation

if __name__ == "__main__":
    # When we perform a simulation, we need to make sure we recover the
    # correct thermal occupation of the states.
    condition = SimulationCondition(
        DIMENSIONLESS_SYSTEM_1D,
        IsotropicSimulationConfig(
            simulation_basis=MomentumSimulationBasis(
                shape=(3,), resolution=(55,), truncation=(3 * 45,)
            ),
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

    diagonal_hamiltonian = linalg.into_diagonal_hermitian(condition.hamiltonian)
    target_occupation = get_eigenvalue_occupation_hermitian(
        diagonal_hamiltonian, condition.config.temperature
    )
    fig, ax, line = plot.basis_against_array(target_occupation)
    ax.set_title("Thermal occupation of the states")
    ax.set_xlabel("Energy /J")
    ax.set_ylabel("Occupation Probability")
    line.set_marker("x")
    fig.show()

    # Now we can test the thermal occupation of the states in our
    # periodic environment.
    times = spaced_time_basis(n=1000, dt=1 * np.pi * hbar)
    states = run_simulation(condition, times)
    states = state.normalize_states(states)
    states = states.with_state_basis(diagonal_hamiltonian.basis.inner[0])

    average_occupation, std_occupation = state.get_average_occupations(states)
    average_occupation = array.cast_basis(average_occupation, target_occupation.basis)
    std_occupation = array.cast_basis(std_occupation, target_occupation.basis)
    # We see that the true occupation of the states is close to the
    # expected thermal occupation.
    fig, ax = plot.get_figure()
    _, _, line = plot.basis_against_array(target_occupation, ax=ax)
    _, _, line = plot.basis_against_array(
        average_occupation, y_error=std_occupation, ax=ax
    )
    ax.set_title("True occupation of the states")
    ax.set_xlabel("Energy /J")
    ax.set_ylabel("Occupation Probability")
    line.set_marker("x")
    fig.show()

    # Just how close are the fitted temperatures to the actual temperatures?
    # We can investigate this by fitting the thermal occupation of the states
    # to the Boltzmann distribution. The error in the fitted temperature
    # is best shown by a plot in log scale.
    info = TemperatureFitInfo(condition.temperature)
    implied_temperature = (
        TemperatureFitMethod()
        .get_fit_from_data(average_occupation, info, y_error=std_occupation)
        .temperature
    )
    fitted_occupation = get_eigenvalue_occupation_hermitian(
        diagonal_hamiltonian, implied_temperature
    )

    fig, ax = plot.get_figure()
    _, _, line = plot.basis_against_array(fitted_occupation, ax=ax, scale="log")
    line.set_label(f"Fitted ({implied_temperature * Boltzmann:.2e})")
    _, _, line = plot.basis_against_array(target_occupation, ax=ax, scale="log")
    line.set_label(f"Actual ({condition.temperature * Boltzmann:.2e})")
    _, _, line = plot.basis_against_array(
        average_occupation, y_error=std_occupation, ax=ax, scale="log"
    )
    line.set_label("Average")
    line.set_marker("x")
    ax.legend()
    ax.set_title(
        f"True occupation of states (T={condition.temperature * Boltzmann:.2e})"
    )
    ax.set_xlabel("Energy /J")
    ax.set_ylabel("Occupation Probability")
    fig.show()
    input()

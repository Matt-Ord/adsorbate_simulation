from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore lib
from slate import Array, array, plot
from slate_quantum import operator, state

from adsorbate_simulation.constants.system import DIMENSIONLESS_1D_FREE_SYSTEM
from adsorbate_simulation.fit import (
    TemperatureFitInfo,
    TemperatureFitMethod,
)
from adsorbate_simulation.simulate import run_stochastic_simulation
from adsorbate_simulation.system import (
    IsotropicSimulationConfig,
    MomentumSimulationBasis,
    PeriodicCaldeiraLeggettEnvironment,
    SimulationCondition,
)
from adsorbate_simulation.util import (
    get_eigenvalue_occupation_hermitian,
    spaced_time_basis,
)

if __name__ == "__main__":
    # Under what condition does the thermal occupation of the states converge?
    # To investigate this we can fit the thermal occupation of the states
    # to a Boltzmann distribution and observe the implied temperature of the simulation.
    # This implied temperature can then be plotted against the actual temperature
    condition = SimulationCondition(
        DIMENSIONLESS_1D_FREE_SYSTEM,
        IsotropicSimulationConfig(
            simulation_basis=MomentumSimulationBasis(
                shape=(3,), resolution=(55,), truncation=(3 * 45,)
            ),
            environment=PeriodicCaldeiraLeggettEnvironment(_eta=8 * hbar / 3**2),
            temperature=10 * hbar / Boltzmann,
        ),
    )

    times = spaced_time_basis(n=1000, dt=1 * np.pi * hbar)

    diagonal_hamiltonian = operator.into_diagonal_hermitian(condition.hamiltonian)
    target_occupation = get_eigenvalue_occupation_hermitian(
        diagonal_hamiltonian, condition.config.temperature
    )

    temperatures = np.array([1, 2, 6, 8, 10]) / Boltzmann
    implied_temperatures = list[float]()
    for temperature in temperatures:
        condition = condition.with_temperature(temperature)
        states = run_stochastic_simulation(condition, times)

        states = states.with_state_basis(diagonal_hamiltonian.basis.inner[0])
        average_occupation, std_occupation = state.get_average_occupations(states)
        average_occupation = array.cast_basis(
            average_occupation, target_occupation.basis
        )
        std_occupation = array.cast_basis(std_occupation, target_occupation.basis)
        implied_temperatures.append(
            TemperatureFitMethod()
            .get_fit_from_data(
                average_occupation,
                TemperatureFitInfo(temperature),
                y_error=std_occupation,
            )
            .temperature
        )

    target = Array.from_array(temperatures)
    implied = Array.from_array(np.array(implied_temperatures))
    fig, ax = plot.get_figure()
    fig, ax, line = plot.array_against_array(target, implied, ax=ax)
    _, _, line = plot.array_against_array(target, target, ax=ax)
    ax.set_xlim(0, None)
    fig.show()
    plot.wait_for_close()

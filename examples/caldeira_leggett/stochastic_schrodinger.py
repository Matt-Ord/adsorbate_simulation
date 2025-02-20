from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore libary
from slate import basis, plot
from slate.plot import (
    animate_data_over_list_1d_k,
    animate_data_over_list_1d_x,
)
from slate_quantum import dynamics, state

from adsorbate_simulation.constants.system import DIMENSIONLESS_1D_FREE_SYSTEM
from adsorbate_simulation.simulate import run_stochastic_simulation
from adsorbate_simulation.system import (
    CaldeiraLeggettEnvironment,
    IsotropicSimulationConfig,
    MomentumSimulationBasis,
    SimulationCondition,
)
from adsorbate_simulation.util import spaced_time_basis

if __name__ == "__main__":
    # This is a simple example of simulating a harmonic system using the
    # caldeira leggett model.
    # This example simulates an plots a single stochastic realization as
    # calculated under the stochastic Schrodinger equation.
    condition = SimulationCondition(
        DIMENSIONLESS_1D_FREE_SYSTEM,
        IsotropicSimulationConfig(
            simulation_basis=MomentumSimulationBasis(
                shape=(2,), resolution=(55,), truncation=(2 * 45,)
            ),
            environment=CaldeiraLeggettEnvironment(_eta=3 / (hbar * 2**2)),
            temperature=10 / Boltzmann,
            target_delta=1e-3,
        ),
    )

    # We simulate the system using the stochastic Schrodinger equation.
    # We find a localized stochastic evolution of the wavepacket.
    times = spaced_time_basis(n=20000, dt=0.1 * np.pi * hbar)
    states = run_stochastic_simulation(condition, times)
    states = dynamics.select_realization(states)

    # We start the system in an gaussian state, centered at the origin.
    fig, ax, _ = plot.array_against_axes_1d(states[0, :], measure="abs")
    ax.set_title("Initial State - A Gaussian Wavepacket Centered at the Origin")
    fig.show()

    fig, ax, anim0 = animate_data_over_list_1d_x(states, measure="abs")
    fig.show()
    fig, ax, anim1 = animate_data_over_list_1d_k(states, measure="abs")
    fig.show()

    # Check that the states are normalized - this is an easy way to check that
    # we have a small enough time step.
    normalization = state.all_inner_product(states, states)
    fig, ax, line = plot.array_against_basis(normalization, measure="real")
    ax.set_title("Normalization of the states against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Normalization")
    ylim = ax.get_ylim()
    delta = max(1 - ylim[0], ylim[1] - 1)
    ax.set_ylim(1 - delta, 1 + delta)
    fig.show()

    plot.wait_for_close()

    basis_list = basis.as_index_basis(basis.as_tuple_basis(states.basis)[0])
    for i in basis_list.points:
        s = states[i.item(), :]
        assert np.isclose(1, normalization.as_array(), atol=1e-2)

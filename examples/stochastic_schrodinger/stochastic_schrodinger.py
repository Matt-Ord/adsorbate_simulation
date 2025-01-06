from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore libary
from slate import basis, plot
from slate.plot import (
    animate_data_over_list_1d_k,
    animate_data_over_list_1d_x,
)
from slate_quantum import dynamics, state

from adsorbate_simulation.system import (
    DIMENSIONLESS_SYSTEM_1D,
    IsotropicSimulationConfig,
    MomentumSimulationBasis,
    PeriodicCaldeiraLeggettEnvironment,
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
            simulation_basis=MomentumSimulationBasis(
                shape=(3,), resolution=(45,), truncation=(3 * 35,)
            ),
            environment=PeriodicCaldeiraLeggettEnvironment(_eta=4 / 3**2),
            temperature=10 / Boltzmann,
        ),
    )

    # We simulate the system using the stochastic Schrodinger equation.
    # We find a localized stochastic evolution of the wavepacket.
    times = spaced_time_basis(n=1000, dt=1 * np.pi * hbar)
    states = run_simulation(condition, times)
    states = dynamics.select_realization(states)

    # We start the system in an gaussian state, centered at the origin.
    fig, ax, _ = plot.basis_against_array_1d_x(states[0, :], measure="abs")
    ax.set_title("Initial State - A Gaussian Wavepacket Centered at the Origin")
    fig.show()

    fig, ax, _anim0 = animate_data_over_list_1d_x(states, measure="abs")
    fig.show()
    fig, ax, _anim1 = animate_data_over_list_1d_k(states, measure="abs")
    fig.show()

    # Check that the states are normalized - this is an easy way to check that
    # we have a small enough time step.
    normalization = state.all_inner_product(states, states)
    fig, ax, line = plot.basis_against_array(normalization, measure="real")
    ax.set_title("Normalization of the states against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Normalization")
    ylim = ax.get_ylim()
    delta = max(1 - ylim[0], ylim[1] - 1)
    ax.set_ylim(1 - delta, 1 + delta)
    fig.show()
    input()

    basis_list = basis.as_index_basis(basis.as_tuple_basis(states.basis)[0])
    for i in basis_list.points:
        s = states[i.item(), :]
        assert np.isclose(1, state.normalization(s), atol=1e-2)

    input()

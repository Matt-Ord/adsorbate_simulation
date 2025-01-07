from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore library
from slate import metadata, plot

from adsorbate_simulation.constants.system import DIMENSIONLESS_1D_SYSTEM
from adsorbate_simulation.simulate import run_stochastic_simulation
from adsorbate_simulation.system import (
    IsotropicSimulationConfig,
    MomentumSimulationBasis,
    PeriodicCaldeiraLeggettEnvironment,
    SimulationCondition,
)
from adsorbate_simulation.util import (
    get_gaussian_width,
    get_momentum,
    get_periodic_position,
    spaced_time_basis,
)

if __name__ == "__main__":
    # Gaussian states are a common choice for initial state
    # but what distribution of p,x,sigma should we choose?
    # For a consistent choice, we can run a simulation
    # and fit the resulting states to a Gaussian.
    condition = SimulationCondition(
        DIMENSIONLESS_1D_SYSTEM,
        IsotropicSimulationConfig(
            simulation_basis=MomentumSimulationBasis(
                shape=(3,), resolution=(45,), truncation=(3 * 35,)
            ),
            environment=PeriodicCaldeiraLeggettEnvironment(_eta=4 / 3**2),
            temperature=10 / Boltzmann,
        ),
    )
    times = spaced_time_basis(n=10001, dt=0.1 * np.pi * hbar)
    states = run_stochastic_simulation(condition, times)

    # Using the e^{ikx} operator, we can calculate the position
    # of the wavepacket.
    positions = get_periodic_position(states, axis=0)
    fig, ax, line = plot.basis_against_array(positions, measure="real")
    ax.set_title("Displacement of the wavepacket against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Position (a.u.)")
    delta_x = metadata.volume.fundamental_stacked_delta_x(states.basis.metadata()[1])
    ax.set_ylim(0, delta_x[0][0])
    fig.show()
    # We have a free particle, so the wavepacket is equally likely
    # to be found at any position.
    fig, ax = plot.array_distribution(positions)
    ax.set_title("Distribution of wavepacket position")
    ax.set_xlabel("Position (a.u.)")
    ax.set_ylabel("Probability")
    fig.show()

    # We can also calculate the width of the wavepacket
    # This remains almost constant over the course of the simulation.
    widths = get_gaussian_width(states, axis=0)
    fig, ax, line = plot.basis_against_array(widths, measure="real")
    ax.set_title("Width of the wavepacket against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Width (a.u.)")
    fig.show()
    # The width of the wavepacket oscillates about the equilibrium width
    fig, ax = plot.array_distribution(widths, distribution="normal")
    ax.set_title("Distribution of wavepacket width")
    ax.set_xlabel("Width (a.u.)")
    ax.set_ylabel("Probability")
    fig.show()

    # The simulation is not periodic in momentum space, so we can use
    # the k operator directly to calculate the momentum of the wavepacket.
    momentums = get_momentum(states, axis=0)
    fig, ax, line = plot.basis_against_array(momentums, measure="real")
    ax.set_title("Momentum of the wavepacket against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Momentum (a.u.)")
    delta_k = metadata.volume.fundamental_stacked_delta_k(states.basis.metadata()[1])
    ax.set_ylim(-delta_k[0][0] / 2, delta_k[0][0] / 2)
    fig.show()
    # The distribution of momentum of the wavepacket is centered at zero,
    fig, ax = plot.array_distribution(momentums, distribution="normal")
    ax.set_title("Distribution of wavepacket momentum")
    ax.set_xlabel("Momentum (a.u.)")
    ax.set_ylabel("Probability")
    fig.show()

    input()

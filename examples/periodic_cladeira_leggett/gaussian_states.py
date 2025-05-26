from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore library
from slate_core import metadata, plot
from slate_quantum import operator

from adsorbate_simulation.constants.system import DIMENSIONLESS_1D_SYSTEM
from adsorbate_simulation.simulate import run_stochastic_simulation
from adsorbate_simulation.system import (
    IsotropicSimulationConfig,
    MomentumSimulationBasis,
    PeriodicCaldeiraLeggettEnvironment,
    SimulationCondition,
)
from adsorbate_simulation.util import (
    EtaParameters,
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
                shape=(2,), resolution=(55,), truncation=(2 * 45,)
            ),
            environment=PeriodicCaldeiraLeggettEnvironment(_eta=3 / (hbar * 2**2)),
            temperature=10 / Boltzmann,
            target_delta=1e-3,
        ),
    )
    times = spaced_time_basis(n=100, dt=0.1 * np.pi * hbar)
    states = run_stochastic_simulation(condition, times)

    # Using the e^{ikx} operator, we can calculate the position
    # of the wavepacket.
    positions = operator.measure.all_periodic_x(states, axis=0)
    fig, ax, line = plot.array_against_basis(positions, measure="real")
    ax.set_title("Displacement of the wavepacket against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Position (a.u.)")
    delta_x = metadata.volume.fundamental_stacked_delta_x(
        states.basis.metadata().children[1]
    )
    ax.set_ylim(0, delta_x[0][0])
    fig.show()
    # We have a free particle, so the wavepacket is equally likely
    # to be found at any position.
    fig, ax = plot.array_distribution(positions)
    ax.set_title("Distribution of wavepacket position")
    ax.set_xlabel("Position (a.u.)")
    ax.set_ylabel("Probability")
    ax.set_xlim(0, delta_x[0][0])
    fig.show()

    # We can also calculate the width of the wavepacket
    # This remains almost constant over the course of the simulation.
    widths = operator.measure.all_variance_x(states, axis=0)
    params = EtaParameters.from_condition(condition)
    theoretical_width = params.get_free_particle_variance_x()
    fig, ax, line = plot.array_against_basis(widths, measure="real")
    line = ax.axhline(theoretical_width)
    line.set_color("red")
    line.set_label("Theoretical width")
    ax.set_title("Width of the wavepacket against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Width (a.u.)")
    fig.show()
    # The width of the wavepacket oscillates about the equilibrium width
    fig, ax = plot.array_distribution(widths, distribution="normal")
    line = ax.axvline(theoretical_width)
    line.set_color("red")
    line.set_label("Theoretical width")
    ax.set_title("Distribution of wavepacket width")
    ax.set_xlabel("Width (a.u.)")
    ax.set_ylabel("Probability")
    fig.show()

    # The simulation is not periodic in momentum space, so we can use
    # the k operator directly to calculate the momentum of the wavepacket.
    momentums = operator.measure.all_k(states, axis=0)
    fig, ax, line = plot.array_against_basis(momentums, measure="real")
    ax.set_title("Momentum of the wavepacket against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Momentum (a.u.)")
    delta_k = metadata.volume.fundamental_stacked_delta_k(
        states.basis.metadata().children[1]
    )
    ax.set_ylim(-delta_k[0][0] / 2, delta_k[0][0] / 2)
    fig.show()
    # The distribution of momentum of the wavepacket is centered at zero,
    fig, ax = plot.array_distribution(momentums, distribution="normal")
    ax.set_title("Distribution of wavepacket momentum")
    ax.set_xlabel("Momentum (a.u.)")
    ax.set_ylabel("Probability")
    fig.show()

    plot.wait_for_close()

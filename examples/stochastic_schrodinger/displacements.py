from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore library
from slate import array, plot

from adsorbate_simulation.system import (
    DIMENSIONLESS_SYSTEM_1D,
    IsotropicSimulationConfig,
    MomentumSimulationBasis,
    PeriodicCaldeiraLeggettEnvironment,
    SimulationCondition,
)
from adsorbate_simulation.util import (
    get_free_displacements,
    get_periodic_position,
    get_restored_displacements,
    get_restored_position,
    run_simulation,
    spaced_time_basis,
)

if __name__ == "__main__":
    # An important quantity to consider when simulating a system is the
    # displacement of the wavepacket against time. The wavefunction is periodic
    # in the position basis and so we can't directly measure the displacement
    # of the wavepacket. However, we can extract the displacement from a periodic
    # measure e^{ikx} for some k which is periodic in the simulation basis
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
    times = spaced_time_basis(n=1000, dt=0.1 * np.pi * hbar)
    states = run_simulation(condition, times)

    # The periodic position is the position of the wavepacket
    # in the simulation basis which is periodic.
    fig, ax = plot.get_figure()
    positions = get_periodic_position(states, axis=0)[0, slice(None)]
    _, _, line = plot.basis_against_array(positions, measure="real", ax=ax)
    line.set_label("Periodic position")

    # To calculate the restored position, we can identify the periodic
    # discontinuities in the periodic data to 'unwrap' the wavepacket.
    positions = get_restored_position(states, axis=0)[0, slice(None)]
    _, _, line = plot.basis_against_array(positions, measure="real", ax=ax)
    line.set_label("Restored position")

    ax.set_ylabel("Position (a.u.)")
    ax.set_xlabel("Time /s")
    ax.set_title("Position of the wavepacket against time")
    ax.legend()
    fig.show()

    # Given the position of the wavepacket, we can calculate the displacement
    # of the wavepacket against time.
    fig, ax = plot.get_figure()
    displacements = array.sqrt(
        get_restored_displacements(states, axis=0)[0, slice(None)]
    )
    _, _, line = plot.basis_against_array(displacements, measure="real", ax=ax)
    line.set_label("Restored displacement")

    free_displacements = get_free_displacements(condition, times.metadata())
    _, _, line = plot.basis_against_array(free_displacements, measure="real", ax=ax)
    line.set_label("Free displacement")

    ax.set_title("Displacement of the wavepacket against time")
    ax.set_xlabel("Time /s")
    ax.set_ylabel("Displacement (a.u.)")
    ax.legend()
    fig.show()
    input()

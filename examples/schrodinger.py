from __future__ import annotations

import numpy as np
from scipy.constants import hbar  # type: ignore libary
from slate import FundamentalBasis, array, plot
from slate.metadata import LabelSpacing
from slate.plot import animate_data_over_list_1d_x
from slate_quantum import operator
from slate_quantum.dynamics import solve_schrodinger_equation_decomposition
from slate_quantum.metadata import SpacedTimeMetadata

from adsorbate_simulation.constants.system import DIMENSIONLESS_1D_FREE_SYSTEM
from adsorbate_simulation.system import (
    FundamentalSimulationBasis,
)

if __name__ == "__main__":
    # This is a simple example of plotting the eigenstates of an adsorbate system.
    # Here we create a system in 1D with a repeating cosine potential
    system = DIMENSIONLESS_1D_FREE_SYSTEM
    # We create a basis with 3 unit cells and 100 points per unit cell
    basis = FundamentalSimulationBasis(shape=(3,), resolution=(100,))
    # We add a cosine potential to the system
    system = system.with_potential(system.potential.with_barrier_height(1.0))
    # The new potential has a non-zero barrier height
    potential = system.get_potential(basis)
    fig, ax, line = plot.array_against_axes_1d(array.as_outer_array(potential))
    line.set_linestyle("--")
    line.set_alpha(0.5)
    line.set_color("black")
    line.set_linewidth(2)

    hamiltonian = system.get_hamiltonian(basis)
    eigenstates = (
        operator.into_diagonal_hermitian(hamiltonian).basis.inner[1].eigenvectors
    )
    # We start the system in an eigenstate of the new potential
    # if we evolve it in time it remains in the same state
    # but the global phase changes
    initial_state = eigenstates[0, :]
    times = FundamentalBasis(
        SpacedTimeMetadata(60, spacing=LabelSpacing(delta=8 * np.pi * hbar))
    )
    states = solve_schrodinger_equation_decomposition(initial_state, times, hamiltonian)
    fig, _, _anim0 = animate_data_over_list_1d_x(states, measure="real", ax=ax.twinx())
    fig.show()

    plot.wait_for_close()

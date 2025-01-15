from __future__ import annotations

from slate import array, plot
from slate_quantum import state

from adsorbate_simulation.constants.system import LI_CU_111_1D_SYSTEM
from adsorbate_simulation.system import FundamentalSimulationBasis

if __name__ == "__main__":
    # This is a simple example of plotting a potential.
    # Here we create a system in 1D with a repeating cosine potential
    system = LI_CU_111_1D_SYSTEM
    # We create a basis with 3 unit cells and 100 points per unit cell
    basis = FundamentalSimulationBasis(shape=(3,), resolution=(100,))

    # We get the potential for the system and plot it
    # The potential is an Operator which is diagonal in position basis
    # Here .as_outer() is used to get the potential as an array of points along the diagonal
    potential = system.get_potential(basis)
    fig, ax, line = plot.array_against_axes_1d(array.as_outer_array(potential))
    line.set_color("black")
    line.set_linestyle("--")
    line.set_alpha(0.5)
    line.set_linewidth(2)

    initial_state = state.build_coherent_state(
        system.get_hamiltonian(basis).basis.metadata()[0], (2e-10,), (0,), (1e-11,)
    )
    _, _, _ = plot.array_against_axes_1d(initial_state, measure="abs", ax=ax.twinx())

    fig.show()
    plot.wait_for_close()

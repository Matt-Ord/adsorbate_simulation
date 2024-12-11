from __future__ import annotations

from slate import array
from slate.plot import plot_data_1d_x
from slate_quantum import state

from adsorbate_simulation.system import LI_CU_SYSTEM_1D, SimulationBasis

if __name__ == "__main__":
    # This is a simple example of plotting a potential.
    # Here we create a system in 1D with a repeating cosine potential
    system = LI_CU_SYSTEM_1D
    # We create a basis with 3 unit cells and 100 points per unit cell
    basis = SimulationBasis((3,), (100,))

    # We get the potential for the system and plot it
    # The potential is an Operator which is diagonal in position basis
    # Here .as_outer() is used to get the potential as an array of points along the diagonal
    potential = system.get_potential(basis)
    fig, ax, line = plot_data_1d_x(array.as_outer_array(potential))
    line.set_color("black")
    line.set_linestyle("--")
    line.set_alpha(0.5)
    line.set_linewidth(2)

    initial_state = state.build_coherent_state(
        system.get_hamiltonian(basis).basis.metadata()[0], (2e-10,), (0,), (1e-11,)
    )
    _, _, _ = plot_data_1d_x(initial_state, measure="abs", ax=ax.twinx())

    fig.show()
    input()

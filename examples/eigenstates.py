from __future__ import annotations

from slate import array, plot
from slate_quantum import operator

from adsorbate_simulation.system import (
    DIMENSIONLESS_SYSTEM_1D,
    FundamentalSimulationBasis,
)

if __name__ == "__main__":
    # This is a simple example of plotting the eigenstates of an adsorbate system.
    # Here we create a system in 1D with a repeating cosine potential
    system = DIMENSIONLESS_SYSTEM_1D
    # We create a basis with 3 unit cells and 100 points per unit cell
    basis = FundamentalSimulationBasis(shape=(3,), resolution=(100,))

    # Get the hamiltonian for the system
    hamiltonian = system.get_hamiltonian(basis)
    # TODO: we want to make this more natural ...  # noqa: FIX002
    eigenstates = (
        operator.into_diagonal_hermitian(hamiltonian).basis.inner[1].eigenvectors
    )
    # The eigenstates of this system are the momentum states
    fig, _, _ = plot.basis_against_array_1d_k(eigenstates[0])
    fig.show()

    # But what if we were to add a cosine potential to the system?
    system = system.with_potential(system.potential.with_barrier_height(1.0))
    # The new potential has a non-zero barrier height
    potential = system.get_potential(basis)
    fig, ax, _ = plot.basis_against_array_1d_x(array.as_outer_array(potential))
    hamiltonian = system.get_hamiltonian(basis)
    eigenstates = (
        operator.into_diagonal_hermitian(hamiltonian).basis.inner[1].eigenvectors
    )
    # The eigenstates of this system are plotted in position basis
    # We can see that the state is localized around the minima of the potential
    _, _, line = plot.basis_against_array_1d_x(eigenstates[0], ax=ax.twinx())
    line.set_color("C1")
    fig.show()

    input()

from __future__ import annotations

from matplotlib.scale import SymmetricalLogScale
from scipy.constants import hbar  # type: ignore library
from slate_core import array, plot
from slate_quantum import operator

from adsorbate_simulation.constants.system import (
    DIMENSIONLESS_1D_SYSTEM,
)
from adsorbate_simulation.system import (
    FundamentalSimulationBasis,
    HarmonicPotential,
    PositionSimulationBasis,
)

if __name__ == "__main__":
    # This is a simple example of plotting the eigenstates of an adsorbate system.
    # Here we create a system in 1D with a repeating cosine potential
    system = DIMENSIONLESS_1D_SYSTEM
    # We create a basis with 3 unit cells and 100 points per unit cell
    basis = FundamentalSimulationBasis(shape=(3,), resolution=(100,))

    # Get the hamiltonian for the system
    hamiltonian = system.get_hamiltonian(basis)
    eigenstates = operator.get_eigenstates_hermitian(hamiltonian)
    # The eigenstates of this system are the momentum states
    fig, ax, _ = plot.array_against_axes_1d_k(eigenstates[0, :], measure="abs")
    ax.set_title("Eigenstates of a free system")
    fig.show()

    # But what if we were to add a cosine potential to the system?
    system = system.with_potential(system.potential.with_barrier_height(5.0))
    # The new potential has a non-zero barrier height
    potential = system.get_potential(basis)
    fig, ax, _ = plot.array_against_axes_1d(array.extract_diagonal(potential))
    hamiltonian = system.get_hamiltonian(basis)
    eigenstates = operator.get_eigenstates_hermitian(hamiltonian)
    # The eigenstates of this system are plotted in position basis
    # We can see that the state is localized around the minima of the potential
    _, ax, line = plot.array_against_axes_1d(
        eigenstates[0, :], measure="abs", ax=ax.twinx()
    )
    ax.set_title("Eigenstates of a Cosine system")
    line.set_color("C1")
    fig.show()

    # Similarly, what are the eigenstates of a harmonic system?
    system = system.with_potential(HarmonicPotential(frequency=10 / hbar))
    basis = PositionSimulationBasis(shape=(1,), resolution=(100,))
    potential = system.get_potential(basis)
    fig, ax, _ = plot.array_against_axes_1d(array.extract_diagonal(potential))
    ax2 = ax.twinx()
    # We want to use hard wall boundary conditions, choosing
    # a suitable truncation for the basis. We find that 50
    # is enough to approximate the first 4 eigenstates,
    # but for 25 the states are visibly truncated.
    for i, truncation in enumerate([25, 30, 50, 80, 100]):
        basis = PositionSimulationBasis(
            shape=(1,),
            resolution=(100,),
            truncation=(truncation,),
            offset=(50 - truncation // 2,),
        )
        hamiltonian = system.get_hamiltonian(basis)
        hamiltonian = hamiltonian.with_basis(basis.get_operator_basis(system.cell))
        eigenstates = operator.get_eigenstates_hermitian(hamiltonian)
        # The eigenstates of this system are plotted in position basis
        # We can see that the state is localized around the minima of the potential
        _, _, line = plot.array_against_axes_1d(
            eigenstates[24, :], measure="abs", ax=ax2
        )
        line.set_label(f"Truncation {truncation}")
        line.set_color(f"C{i}")
    ax2.set_yscale(SymmetricalLogScale(None, linthresh=1e-3))
    ax2.legend()
    ax.set_title("Eigenstates of a Harmonic system")
    fig.show()

    plot.wait_for_close()

from __future__ import annotations

from scipy.constants import Boltzmann, hbar  # type: ignore unknown
from slate import linalg, plot

from adsorbate_simulation.constants import (
    DIMENSIONLESS_1D_SYSTEM,
)
from adsorbate_simulation.system import (
    CaldeiraLeggettEnvironment,
    HarmonicPotential,
    IsotropicSimulationConfig,
    PositionSimulationBasis,
    SimulationCondition,
)
from adsorbate_simulation.util import (
    get_eigenvalue_occupation_hermitian,
)

if __name__ == "__main__":
    # When we perform a simulation, we need to make sure we correctly
    # truncate the basis to properly represent the states which would contribute to
    # the dynamics
    # As a back-of-the-envelope calculation, we
    # need to make sure we have states up to
    # KT ~ V(x),
    # for
    # temperature of 10 / K
    # V(x) = 0.5 * (20 / hbar)**2 * x^2
    # we need x = hbar / sqrt(20)
    # we have delta_x = 2 * np.pi * hbar
    system = DIMENSIONLESS_1D_SYSTEM.with_potential(
        HarmonicPotential(frequency=20 / hbar)
    )
    condition = SimulationCondition(
        system,
        IsotropicSimulationConfig(
            simulation_basis=PositionSimulationBasis(
                shape=(1,),
                resolution=(100,),
                offset=((50 - 80) // 2,),
                truncation=(80,),
            ),
            environment=CaldeiraLeggettEnvironment(_eta=3 / (hbar * 2**2)),
            temperature=10 / Boltzmann,
            target_delta=1e-3,
        ),
    )

    # For a system in thermal equilibrium, the probability of a state
    # being occupied is given by the Boltzmann distribution.
    # We need to make sure to include enough states such that the
    # occupation of the highest energy state is negligible.
    # By plotting the occupation of the states, we can verify that
    # 25 states is enough resolution for this system.
    diagonal_hamiltonian = linalg.into_diagonal_hermitian(condition.hamiltonian)
    target_occupation = get_eigenvalue_occupation_hermitian(
        diagonal_hamiltonian, condition.config.temperature
    )
    fig, ax, line = plot.array_against_basis(target_occupation)
    ax.set_title("Thermal occupation of the states")
    ax.set_xlabel("Energy /J")
    ax.set_ylabel("Occupation Probability")
    ax.set_yscale("log")
    line.set_marker("x")
    fig.show()
    plot.wait_for_close()

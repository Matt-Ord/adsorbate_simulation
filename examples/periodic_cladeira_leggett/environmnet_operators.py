from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore libary
from slate_core import array, basis, plot

from adsorbate_simulation.constants.system import DIMENSIONLESS_1D_SYSTEM
from adsorbate_simulation.system import (
    IsotropicSimulationConfig,
    MomentumSimulationBasis,
    PeriodicCaldeiraLeggettEnvironment,
    SimulationCondition,
)

if __name__ == "__main__":
    # This is a simple example of using environment operators in the stochastic schrodinger
    # equation.
    # First we create a simulation condition for a free system in 1D.
    condition = SimulationCondition(
        DIMENSIONLESS_1D_SYSTEM,
        IsotropicSimulationConfig(
            simulation_basis=MomentumSimulationBasis(
                shape=(3,), resolution=(45,), truncation=(3 * 35,)
            ),
            environment=PeriodicCaldeiraLeggettEnvironment(_eta=2 * hbar**2 / 2**2),
            temperature=10 * hbar / Boltzmann,
        ),
    )
    # The Hamiltonian is diagonal in momentum basis
    hamiltonian = condition.hamiltonian.with_basis(condition.operator_basis)
    hamiltonian_flat = array.flatten(hamiltonian)
    hamiltonian_flat = array.as_upcast_basis(
        hamiltonian_flat, hamiltonian_flat.basis.metadata()
    )
    fig, ax, _ = plot.array_against_axes_2d_k(hamiltonian_flat, measure="real")
    ax.set_title("Hamiltonian in Momentum Space")
    fig.show()

    # For the Periodic Caldeiria Leggett environment, we have two environment operators.
    # This generates isotropic periodic noise which matches classical friction.
    environment_operators = condition.get_environment_operators()
    for eigenvalue, operator in zip(
        np.sqrt(environment_operators.basis.metadata().children[0].values),
        environment_operators,
        strict=False,
    ):
        fig, ax, _ = plot.array_against_axes_1d(
            array.extract_diagonal(operator) * eigenvalue, measure="real"
        )
        _, _, line = plot.array_against_axes_1d(
            array.extract_diagonal(operator) * eigenvalue, measure="imag", ax=ax
        )
        ax.set_title("Environment Operator")
        fig.show()

    # For a thermal simulation, we must use temperature corrected operators.
    # These are no longer diagonal in position basis. To understand the effect
    # of this correction, we can plot the corrected operators in momentum space
    # and compare them to the original operators.
    original_operator = environment_operators[0, :]
    corrected_operator = condition.temperature_corrected_operators[0, :].with_basis(
        condition.operator_basis
    )
    # The original operator is a simple scatter from k -> k + \kappa
    # Each scatter has the same probability, and we only see occupation of
    # states which k' = k + \kappa
    original_operator = original_operator.with_basis(
        basis.from_metadata(
            original_operator.basis.metadata(), is_dual=original_operator.basis.is_dual
        ).upcast()
    )
    operator_flat = array.flatten(original_operator)
    operator_flat = array.as_upcast_basis(operator_flat, operator_flat.basis.metadata())
    fig, ax, _ = plot.array_against_axes_2d_k(operator_flat, measure="abs")
    ax.set_title("Original Environment Operator in Momentum Space")
    fig.show()
    # When we apply the temperature correction, the operator scatters more strongly
    # if the initial state has a higher energy than the final state. In this case we
    # are scatting by a negative k, so we see this increase in the positive k region.
    fig, ax, _ = plot.array_against_axes_2d_k(operator_flat, measure="abs")
    ax.set_title("Temperature Corrected Environment Operator in Momentum Space")
    fig.show()
    fig, ax, _ = plot.array_against_axes_2d(operator_flat, measure="real")
    ax.set_title("Temperature Corrected Environment Operator in Position Space")
    fig.show()

    # The equivalent plot for the second operator, we see an increase in the negative k region.
    # In both cases we are more likely to scatter to states with lower energy.
    corrected_operator = condition.temperature_corrected_operators[1, :].with_basis(
        condition.operator_basis
    )
    operator_flat = array.flatten(corrected_operator)
    operator_flat = array.as_upcast_basis(operator_flat, operator_flat.basis.metadata())
    fig, ax, _ = plot.array_against_axes_2d_k(operator_flat, measure="abs")
    ax.set_title("Second Temperature Corrected Environment Operator in Momentum Space")
    fig.show()

    plot.wait_for_close()

from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann  # type: ignore libary
from slate import array, basis, plot

from adsorbate_simulation.system import (
    DIMENSIONLESS_SYSTEM_1D,
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
        DIMENSIONLESS_SYSTEM_1D,
        IsotropicSimulationConfig(
            simulation_basis=MomentumSimulationBasis(shape=(4,), resolution=(25,)),
            environment=PeriodicCaldeiraLeggettEnvironment(_eta=4 / 3**2),
            temperature=1 / Boltzmann,
        ),
    )
    # The Hamiltonian is diagonal in momentum basis
    hamiltonian = condition.hamiltonian.with_basis(
        condition.config.simulation_basis.get_operator_basis(condition.system.cell)
    )
    fig, ax, _ = plot.basis_against_array_2d_k(
        array.flatten(hamiltonian), measure="real"
    )
    fig.show()

    # For the Periodic Caldeiria Leggett environment, we have two environment operators.
    # This generates isotropic periodic noise which matches classical friction.
    environment_operators = condition.get_environment_operators()
    for eigenvalue, operator in zip(
        np.sqrt(environment_operators.basis.metadata()[0].values),
        environment_operators,
        strict=False,
    ):
        fig, ax, _ = plot.basis_against_array_1d_x(
            array.as_diagonal_array(operator) * eigenvalue, measure="real"
        )
        _, _, line = plot.basis_against_array_1d_x(
            array.as_diagonal_array(operator) * eigenvalue, measure="imag", ax=ax
        )
        ax.set_title("Environment Operator")
        fig.show()

    # For a thermal simulation, we must use temperature corrected operators.
    # These are no longer diagonal in position basis. To understand the effect
    # of this correction, we can plot the corrected operators in momentum space
    # and compare them to the original operators.
    original_operator = environment_operators[0, :]
    corrected_operator = condition.temperature_corrected_operators[0, :]
    # The original operator is a simple scatter from k -> k + \kappa
    # Each scatter has the same probability, and we only see occupation of
    # states which k' = k + \kappa
    original_operator = original_operator.with_basis(
        basis.from_metadata(
            original_operator.basis.metadata(), is_dual=original_operator.basis.is_dual
        )
    )
    fig, ax, _ = plot.basis_against_array_2d_k(
        array.flatten(original_operator), measure="abs"
    )
    ax.set_title("Original Environment Operator in Momentum Space")
    fig.show()
    # When we apply the temperature correction, the operator scatters more strongly
    # if the initial state has a higher energy than the final state. In this case we
    # are scatting by a negative k, so we see this increase in the positive k region.
    fig, ax, _ = plot.basis_against_array_2d_k(
        array.flatten(corrected_operator), measure="abs"
    )
    ax.set_title("Temperature Corrected Environment Operator in Momentum Space")
    fig.show()
    fig, ax, _ = plot.basis_against_array_2d_x(
        array.flatten(corrected_operator), measure="real"
    )
    ax.set_title("Temperature Corrected Environment Operator in Position Space")
    fig.show()

    # The equivalent plot for the second operator, we see an increase in the negative k region.
    # In both cases we are more likely to scatter to states with lower energy.
    corrected_operator = condition.temperature_corrected_operators[1, :]
    fig, ax, _ = plot.basis_against_array_2d_k(
        array.flatten(corrected_operator), measure="abs"
    )
    ax.set_title("Second Temperature Corrected Environment Operator in Momentum Space")
    fig.show()
    input()

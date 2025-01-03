from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import Boltzmann  # type: ignore stubs
from slate import (
    Array,
    BasisMetadata,
    FundamentalBasis,
    StackedMetadata,
    array,
    metadata,
)
from slate.linalg import into_diagonal_hermitian
from slate.metadata import (
    AxisDirections,
    ExplicitLabeledMetadata,
    LabelSpacing,
    SpacedLengthMetadata,
)
from slate_quantum import StateList, operator, state
from slate_quantum.metadata import (
    SpacedTimeMetadata,
)

if TYPE_CHECKING:
    from slate.metadata import Metadata2D


def get_thermal_occupation(
    energies: np.ndarray[Any, np.dtype[np.floating]], temperature: float
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Get the occupation of the eigenvalues of a Hermitian operator."""
    occupation = np.exp(-energies / (Boltzmann * temperature))
    occupation /= occupation.sum()
    return occupation


def get_eigenvalue_occupation_hermitian[M: BasisMetadata](
    array: Array[Metadata2D[M, M, Any], np.complexfloating], temperature: float
) -> Array[ExplicitLabeledMetadata[np.floating], np.floating]:
    """Get the occupation of the eigenvalues of a Hermitian operator."""
    diagonal = into_diagonal_hermitian(array)
    energies = np.sort(np.abs(diagonal.raw_data))
    energy_basis = FundamentalBasis(ExplicitLabeledMetadata(energies))

    occupation = get_thermal_occupation(energies, temperature)
    return Array(energy_basis, occupation)


def spaced_time_basis(*, n: int, dt: float) -> FundamentalBasis[SpacedTimeMetadata]:
    """Get a Time Basis with a given number of steps between each time step."""
    return FundamentalBasis(SpacedTimeMetadata(n, spacing=LabelSpacing(delta=n * dt)))


def _get_fundamental_scatter[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[M0, StackedMetadata[M1, E]],
    axis: int,
) -> Array[M0, np.complexfloating]:
    r"""Get the scattering operator for a wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    n_dim = len(states.basis.metadata()[1].children)
    n_k = tuple(1 if i == axis else 0 for i in range(n_dim))
    scatter = operator.build.scattering_operator(states.basis.metadata()[1], n_k=n_k)

    states = state.normalize_states(states)
    return operator.expectation_of_each(scatter, states)


def get_gaussian_width[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[M0, StackedMetadata[M1, E]],
    axis: int,
) -> Array[M0, np.floating]:
    r"""Get the width of a Gaussian wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    scatter = _get_fundamental_scatter(states, axis)

    norm = array.abs(scatter)
    dk = metadata.volume.fundamental_stacked_dk(states.basis.metadata()[1])[axis]
    q = np.linalg.norm(dk).item()
    return array.sqrt(array.log(norm) * -(4 / q**2))


def get_periodic_position[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[M0, StackedMetadata[M1, E]],
    axis: int,
) -> Array[M0, np.floating]:
    r"""Get the periodic position coordinate of a wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    scatter = _get_fundamental_scatter(states, axis)

    angle = array.angle(scatter)
    wrapped = array.mod(angle, (2 * np.pi))
    delta_x = metadata.volume.fundamental_stacked_delta_x(states.basis.metadata()[1])
    return wrapped * (np.linalg.norm(delta_x[axis]).item() / (2 * np.pi))


def get_restored_position[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[M0, StackedMetadata[M1, E]],
    axis: int,
) -> Array[M0, np.floating]:
    """Get the restored position coordinate of a wavepacket."""
    scatter = _get_fundamental_scatter(states, axis)

    angle = array.angle(scatter)
    unwrapped = array.unwrap(angle, axis=1)
    delta_x = metadata.volume.fundamental_stacked_delta_x(states.basis.metadata()[1])
    return unwrapped * (np.linalg.norm(delta_x[axis]).item() / (2 * np.pi))


def get_momentum[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[M0, StackedMetadata[M1, E]],
    axis: int,
) -> Array[M0, np.floating]:
    """Get the momentum of a wavepacket."""
    momentum = operator.build.k_operator(states.basis.metadata()[1], idx=axis)

    states = state.normalize_states(states)
    return array.real(operator.expectation_of_each(momentum, states))

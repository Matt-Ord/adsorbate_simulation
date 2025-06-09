from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.signal  # type: ignore stubs
from scipy.constants import Boltzmann, hbar  # type: ignore stubs
from slate_core import (
    Array,
    BasisMetadata,
    FundamentalBasis,
    TupleMetadata,
    array,
    basis,
    metadata,
)
from slate_core.linalg import into_diagonal_hermitian
from slate_core.metadata import (
    AxisDirections,
    Domain,
    EvenlySpacedLengthMetadata,
    ExplicitLabeledMetadata,
)
from slate_quantum import operator
from slate_quantum.metadata import (
    SpacedTimeMetadata,
    TimeMetadata,
)

from adsorbate_simulation.system._potential import HarmonicPotential
from adsorbate_simulation.util._eta import gamma_from_eta

if TYPE_CHECKING:
    from slate_core.array import ArrayWithMetadata
    from slate_quantum.state import StateListWithMetadata

    from adsorbate_simulation.system import (
        SimulationCondition,
    )
    from adsorbate_simulation.system._system import System


def get_thermal_occupation(
    energies: np.ndarray[Any, np.dtype[np.floating]], temperature: float
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Get the occupation of the eigenvalues of a Hermitian operator."""
    occupation = np.exp(-energies / (Boltzmann * temperature))
    occupation /= occupation.sum()
    return occupation


def get_eigenvalue_occupation_hermitian[M: BasisMetadata](
    array: ArrayWithMetadata[
        TupleMetadata[tuple[M, M], None], np.dtype[np.complexfloating]
    ],
    temperature: float,
) -> ArrayWithMetadata[
    ExplicitLabeledMetadata[np.dtype[np.floating]], np.dtype[np.floating]
]:
    """Get the occupation of the eigenvalues of a Hermitian operator."""
    diagonal = into_diagonal_hermitian(array)
    energies = np.sort(np.abs(diagonal.raw_data))
    energy_basis = FundamentalBasis(ExplicitLabeledMetadata(energies))

    occupation = get_thermal_occupation(energies, temperature)
    return Array(energy_basis, occupation)


def spaced_time_basis(*, n: int, dt: float) -> FundamentalBasis[SpacedTimeMetadata]:
    """Get a Time Basis with a given number of steps between each time step."""
    return FundamentalBasis(SpacedTimeMetadata(n, domain=Domain(delta=n * dt)))


def get_free_displacement_rate[M: BasisMetadata, DT: np.floating](
    condition: SimulationCondition[Any, Any],
) -> float:
    """Get the rate of displacement of a free particle."""
    gamma = gamma_from_eta(condition.eta, condition.mass)
    return 2 * Boltzmann * condition.temperature / (gamma * np.sqrt(2))


def _get_variance_x_from_ratio(ratio: complex) -> float:
    return np.abs(ratio) ** 2 / (2 * np.real(ratio))


def _get_variance_p_from_ratio(ratio: complex) -> float:
    return hbar**2 / (2 * np.real(ratio))


def _get_uncertainty_from_ratio(ratio: complex) -> float:
    return _get_variance_x_from_ratio(ratio) * _get_variance_p_from_ratio(ratio)


@dataclass(frozen=True, kw_only=True)
class EtaParameters:
    """A set of dimensionless parameters used to characterize the system."""

    eta_m: float
    eta_omega: float | None = None
    eta_lambda: float

    @staticmethod
    def from_condition(
        condition: SimulationCondition[System[Any], Any],
    ) -> EtaParameters:
        """Get the damping coefficients from the simulation condition."""
        gamma = gamma_from_eta(condition.eta, condition.mass)
        eta_lambda = Boltzmann * condition.temperature / (hbar * gamma)

        potential = condition.system.potential

        if isinstance(potential, HarmonicPotential):
            omega = float(potential.frequency / np.sqrt(condition.mass))
            eta_omega = Boltzmann * condition.temperature / (hbar * omega)
        else:
            eta_omega = None

        return EtaParameters(
            eta_m=Boltzmann * condition.temperature / (hbar**2 / (2 * condition.mass)),
            eta_omega=eta_omega,
            eta_lambda=eta_lambda,
        )

    def get_free_particle_width(self) -> float:
        """Calculate the width of a coherent state for a free particle."""
        return np.sqrt(2 * (np.sqrt(2) - 1) / (4 * self.eta_m))

    def get_high_friction_width(self) -> float:
        """Calculate the width of a coherent state for a free particle with high friction."""
        return np.sqrt(2 * (np.sqrt(2 * self.eta_lambda)) / (2 * self.eta_m))

    def get_low_friction_width(self) -> float:
        """Calculate the width of a coherent state for a free particle."""
        eta_omega = self.eta_omega
        assert eta_omega is not None
        return eta_omega / self.eta_m

    def get_free_particle_ratio(self: EtaParameters) -> complex:
        """Calculate the ratio for a free particle."""
        numerator = np.emath.sqrt(2 + 4 * 1j * self.eta_lambda) - 1
        return numerator.item() / (2 * self.eta_m)

    def get_free_particle_variance_x(self: EtaParameters) -> float:
        """Calculate the variance of a coherent state for a free particle."""
        return _get_variance_x_from_ratio(self.get_free_particle_ratio())

    def get_ratio(self) -> complex:
        """Calculate the ratio for a general set of params."""
        eta_lambda = self.eta_lambda
        eta_m = self.eta_m
        eta_omega = self.eta_omega
        assert eta_omega is not None
        sqrt_term = np.emath.sqrt(
            -4 * eta_lambda**2
            + 16j * eta_lambda * eta_omega**2
            + 1j * eta_lambda
            + 8 * eta_omega**2
        ).item()
        prefactor = (1j * eta_omega) / (eta_m * (eta_lambda - 4 * 1j * eta_omega**2))
        return prefactor * (2 * eta_omega - sqrt_term)

    def get_variance_x(self: EtaParameters) -> float:
        """Calculate the variance of a coherent state for a general set of params."""
        return _get_variance_x_from_ratio(self.get_ratio())

    def get_variance_p(self: EtaParameters) -> float:
        """Calculate the variance of a coherent state for a general set of params."""
        return _get_variance_p_from_ratio(self.get_ratio())

    def get_uncertainty(self: EtaParameters) -> float:
        """Calculate the uncertainty of a coherent state for a general set of params."""
        return _get_uncertainty_from_ratio(self.get_ratio())


def get_harmonic_width(params: EtaParameters) -> float:
    """Calculate the width of a coherent state for a harmonic potential.

    Raises
    ------
    ValueError
        If the harmonic potential frequency is not defined.
    """
    eta_omega = params.eta_omega
    if eta_omega is None:
        msg = "The harmonic potential frequency is not defined."
        raise ValueError(msg)
    return 2 * eta_omega / params.eta_m


def get_free_displacements[M: TimeMetadata](
    condition: SimulationCondition[Any, Any],
    times: M,
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Get the displacement of a free particle."""
    rate = get_free_displacement_rate(condition)
    return Array(basis.from_metadata(times), times.values * rate)


def measure_restored_x[
    M0: BasisMetadata,
    M1: EvenlySpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateListWithMetadata[M0, TupleMetadata[tuple[M1, ...], E]],
    *,
    axis: int,
) -> ArrayWithMetadata[M0, np.dtype[np.floating]]:
    """Get the restored position coordinate of a wavepacket."""
    periodic_x = operator.measure.all_periodic_x(states, axis=axis)

    delta_x = metadata.volume.fundamental_stacked_delta_x(
        states.basis.metadata().children[1]
    )
    factor = 2 * np.pi / np.linalg.norm(delta_x[axis]).item()

    return (
        array.unwrap((periodic_x * factor).as_type(np.float64), axis=1) * (1 / factor)
    ).as_type(np.float64)


def _calculate_total_offsset_multiplications_real(
    lhs: np.ndarray[Any, np.dtype[Any]],
    rhs: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Calculate sum_i^N-i A_i B_i+N for all N.

    Parameters
    ----------
    lhs : np.ndarray[Any, np.dtype[np.float64]]
    rhs : np.ndarray[Any, np.dtype[np.float64]]

    Returns
    -------
    np.ndarray[Any, np.dtype[np.float64]]
    """
    return scipy.signal.correlate(lhs, rhs, mode="full")[lhs.size - 1 :]  # type: ignore unknown


def get_restored_displacements[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: EvenlySpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateListWithMetadata[
        TupleMetadata[tuple[M0, M1], None], TupleMetadata[tuple[M2, ...], E]
    ],
    *,
    axis: int,
) -> ArrayWithMetadata[TupleMetadata[tuple[M0, M1], None], np.dtype[np.floating]]:
    """Get the restored displacement of a wavepacket."""
    positions = array.as_fundamental_basis(measure_restored_x(states, axis=axis))
    squared = array.square(positions).as_array()
    total = np.cumsum(squared + squared[:, ::-1], axis=1)[:, ::-1]

    stacked = positions.as_array()
    convolution = np.apply_along_axis(
        lambda m: _calculate_total_offsset_multiplications_real(m, m),
        axis=1,
        arr=stacked,
    )
    size = positions.basis.fundamental_size
    squared_diff = (total - 2 * convolution) / np.arange(1, size + 1)[::-1]
    return Array(positions.basis, squared_diff)


def _calculate_total_offsset_multiplications_complex(
    lhs: np.ndarray[Any, np.dtype[np.complex128]],
    rhs: np.ndarray[Any, np.dtype[np.complex128]],
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """Calculate sum_i^N-i A_i B_i+N for all N.

    Parameters
    ----------
    lhs : np.ndarray[Any, np.dtype[np.float64]]
    rhs : np.ndarray[Any, np.dtype[np.float64]]

    Returns
    -------
    np.ndarray[Any, np.dtype[np.float64]]

    """
    re_re = _calculate_total_offsset_multiplications_real(np.real(lhs), np.real(rhs))
    re_im = _calculate_total_offsset_multiplications_real(np.real(lhs), np.imag(rhs))
    im_re = _calculate_total_offsset_multiplications_real(np.imag(lhs), np.real(rhs))
    im_im = _calculate_total_offsset_multiplications_real(np.imag(lhs), np.imag(rhs))
    return re_re - im_im + 1j * (re_im + im_re)


def get_restored_scatter[
    M0: BasisMetadata,
    M1: EvenlySpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateListWithMetadata[M0, TupleMetadata[tuple[M1, ...], E]],
    k: tuple[float, ...],
) -> ArrayWithMetadata[M0, np.dtype[np.complexfloating]]:
    r"""Get the restored scattering operator for a wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}.
    """
    raise NotImplementedError


def get_restored_isf[
    M0: BasisMetadata,
    M1: EvenlySpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateListWithMetadata[M0, TupleMetadata[tuple[M1, ...], E]],
    k: tuple[float, ...],
) -> ArrayWithMetadata[M0, np.dtype[np.complexfloating]]:
    """Get the restored displacement of a wavepacket."""
    scatter = get_restored_scatter(states, k)

    # convolution_j = \sum_i^N-j e^(ik.x_i+j) e^(-ik.x_i)
    convolution = np.apply_along_axis(
        lambda m: _calculate_total_offsset_multiplications_complex(np.conj(m), m),
        axis=1,
        arr=scatter.raw_data.reshape(1, -1),
    )
    size = scatter.basis.fundamental_size
    isf = (convolution) / np.arange(1, size + 1)[::-1]
    return Array(basis.as_fundamental(scatter.basis), isf)

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import Boltzmann  # type: ignore stubs
from slate import Array, BasisMetadata, FundamentalBasis
from slate.basis import TruncatedBasis, Truncation
from slate.linalg import into_diagonal_hermitian
from slate.metadata import LabelSpacing
from slate_quantum.metadata import (
    EigenvalueMetadata,
    SpacedTimeMetadata,
    eigenvalue_basis,
)

if TYPE_CHECKING:
    from slate.metadata import Metadata2D


def get_eigenvalue_occupation_hermitian[M: BasisMetadata](
    array: Array[Metadata2D[M, M, Any], np.complex128], temperature: float
) -> Array[EigenvalueMetadata, np.float64]:
    """Get the occupation of the eigenvalues of a Hermitian operator."""
    diagonal = into_diagonal_hermitian(array)
    eigenvalues = eigenvalue_basis(np.sort(np.abs(diagonal.raw_data)))

    occupation = np.exp(-eigenvalues.metadata().values / (Boltzmann * temperature))
    occupation /= occupation.sum()
    return Array(eigenvalues, occupation)


def spaced_time_basis(
    *, n: int, step: int = 1000, dt: float
) -> TruncatedBasis[SpacedTimeMetadata, np.generic]:
    """Get a Time Basis with a given number of steps between each time step."""
    return TruncatedBasis(
        Truncation(n, step, 0),
        FundamentalBasis(
            SpacedTimeMetadata(n * step, spacing=LabelSpacing(delta=n * dt))
        ),
    )

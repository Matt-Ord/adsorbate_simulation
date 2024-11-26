from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from slate.basis import Basis, FundamentalBasis, tuple_basis
from slate.metadata import (
    AxisDirections,
    LabelSpacing,
    SpacedLengthMetadata,
    SpacedVolumeMetadata,
)

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class SimulationBasis:
    """The underlying basis for a simulation."""

    shape: tuple[int, ...]
    """The number of repeated units."""
    resolution: tuple[int, ...]
    """The number of states per repeated unit in each direction."""

    def with_resolution(self, resolution: tuple[int, ...]) -> SimulationBasis:
        """Create a new basis with a different resolution."""
        return SimulationBasis(self.shape, resolution)

    def with_shape(self, shape: tuple[int, ...]) -> SimulationBasis:
        """Create a new basis with a different shape."""
        return SimulationBasis(shape, self.resolution)

    def get_repeat_basis(
        self, lengths: tuple[float, ...], directions: AxisDirections
    ) -> Basis[SpacedVolumeMetadata, np.complex128]:
        """Get the basis for the repeat cell of the simulation."""
        return tuple_basis(
            tuple(
                FundamentalBasis(
                    SpacedLengthMetadata(r, spacing=LabelSpacing(delta=delta))
                )
                for (r, delta) in zip(self.resolution, lengths)
            ),
            directions,
        )

    def get_fundamental_basis(
        self, lengths: tuple[float, ...], directions: AxisDirections
    ) -> Basis[SpacedVolumeMetadata, np.complex128]:
        """Get the fundamental basis for the simulation."""
        return tuple_basis(
            tuple(
                FundamentalBasis(
                    SpacedLengthMetadata(s * r, spacing=LabelSpacing(delta=delta * s))
                )
                for (s, r, delta) in zip(self.shape, self.resolution, lengths)
            ),
            directions,
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from slate import StackedMetadata, basis
from slate.metadata import (
    AxisDirections,
    LabelSpacing,
    SpacedLengthMetadata,
    SpacedVolumeMetadata,
)

if TYPE_CHECKING:
    import numpy as np
    from slate.basis import Basis


@dataclass(frozen=True, kw_only=True)
class SimulationCell:
    """Represents a cell in a simulation."""

    lengths: tuple[float, ...]
    directions: AxisDirections

    def with_lengths(self, lengths: tuple[float, ...]) -> SimulationCell:
        """Create a new cell with different lengths."""
        return type(self)(lengths=lengths, directions=self.directions)

    def with_directions(self, directions: AxisDirections) -> SimulationCell:
        """Create a new cell with different directions."""
        return type(self)(
            lengths=self.lengths,
            directions=directions,
        )


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

    def get_repeat_metadata(self, cell: SimulationCell) -> SpacedVolumeMetadata:
        """Get the metadata for the repeat cell of the simulation."""
        return StackedMetadata(
            tuple(
                SpacedLengthMetadata(r, spacing=LabelSpacing(delta=delta))
                for r, delta in zip(self.resolution, cell.lengths)
            ),
            cell.directions,
        )

    def get_repeat_basis(
        self, cell: SimulationCell
    ) -> Basis[SpacedVolumeMetadata, np.complex128]:
        """Get the basis for the repeat cell of the simulation."""
        return basis.from_metadata(self.get_repeat_metadata(cell))

    def get_fundamental_metadata(self, cell: SimulationCell) -> SpacedVolumeMetadata:
        """Get the metadata for the fundamental cell of the simulation."""
        return StackedMetadata(
            tuple(
                SpacedLengthMetadata(s * r, spacing=LabelSpacing(delta=delta * s))
                for (s, r, delta) in zip(self.shape, self.resolution, cell.lengths)
            ),
            cell.directions,
        )

    def get_fundamental_basis(
        self, cell: SimulationCell
    ) -> Basis[SpacedVolumeMetadata, np.complex128]:
        """Get the fundamental basis for the simulation."""
        return basis.from_metadata(self.get_fundamental_metadata(cell))

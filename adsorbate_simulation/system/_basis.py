from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from slate import StackedMetadata, TupleBasis, basis, tuple_basis
from slate.metadata import (
    AxisDirections,
    LabelSpacing,
    SpacedLengthMetadata,
    SpacedVolumeMetadata,
)
from slate_quantum import metadata

if TYPE_CHECKING:
    import numpy as np
    from slate.basis import Basis
    from slate_quantum.metadata import RepeatedVolumeMetadata


@dataclass(frozen=True, kw_only=True)
class SimulationCell:
    """Represents a cell in a simulation."""

    lengths: tuple[float, ...]
    directions: AxisDirections

    def __post_init__(self) -> None:
        assert len(self.lengths) == len(self.directions.vectors)

    def with_lengths(self, lengths: tuple[float, ...]) -> SimulationCell:
        """Create a new cell with different lengths."""
        return type(self)(lengths=lengths, directions=self.directions)

    def with_directions(self, directions: AxisDirections) -> SimulationCell:
        """Create a new cell with different directions."""
        return type(self)(
            lengths=self.lengths,
            directions=directions,
        )


@dataclass(frozen=True, kw_only=True)
class SimulationBasis(ABC):
    """The underlying basis for a simulation."""

    shape: tuple[int, ...]
    """The number of repeated units."""
    resolution: tuple[int, ...]
    """The number of states per repeated unit in each direction."""

    @abstractmethod
    def with_resolution(self, resolution: tuple[int, ...]) -> SimulationBasis:
        """Create a new basis with a different resolution."""

    @abstractmethod
    def with_shape(self, shape: tuple[int, ...]) -> SimulationBasis:
        """Create a new basis with a different shape."""

    def get_repeat_metadata(self, cell: SimulationCell) -> SpacedVolumeMetadata:
        """Get the metadata for the repeat cell of the simulation."""
        return StackedMetadata(
            tuple(
                SpacedLengthMetadata(r, spacing=LabelSpacing(delta=delta))
                for r, delta in zip(self.resolution, cell.lengths, strict=False)
            ),
            cell.directions,
        )

    def get_repeat_basis(
        self, cell: SimulationCell
    ) -> Basis[SpacedVolumeMetadata, np.complex128]:
        """Get the basis for the repeat cell of the simulation."""
        return basis.from_metadata(self.get_repeat_metadata(cell))

    def get_fundamental_metadata(self, cell: SimulationCell) -> RepeatedVolumeMetadata:
        """Get the metadata for the fundamental cell of the simulation."""
        return metadata.repeat_volume_metadata(
            self.get_repeat_metadata(cell), self.shape
        )

    def get_fundamental_basis(
        self, cell: SimulationCell
    ) -> TupleBasis[SpacedLengthMetadata, AxisDirections, np.complex128]:
        """Get the fundamental basis for the simulation."""
        return basis.from_metadata(self.get_fundamental_metadata(cell))

    def get_operator_basis(
        self, cell: SimulationCell
    ) -> basis.TupleBasis2D[
        np.complexfloating,
        Basis[SpacedVolumeMetadata, np.complex128],
        Basis[SpacedVolumeMetadata, np.complex128],
        None,
    ]:
        """Get the basis for the simulation."""
        state_basis = self.get_simulation_basis(cell)
        return tuple_basis((state_basis, state_basis.dual_basis()))

    @abstractmethod
    def get_simulation_basis(
        self, cell: SimulationCell
    ) -> Basis[SpacedVolumeMetadata, np.complex128]:
        """Get the basis for the simulation."""


class FundamentalSimulationBasis(SimulationBasis):
    """The fundamental basis for a simulation."""

    @override
    def with_resolution(
        self, resolution: tuple[int, ...]
    ) -> FundamentalSimulationBasis:
        return type(self)(shape=self.shape, resolution=resolution)

    @override
    def with_shape(self, shape: tuple[int, ...]) -> FundamentalSimulationBasis:
        return type(self)(shape=shape, resolution=self.resolution)

    @override
    def get_simulation_basis(
        self, cell: SimulationCell
    ) -> Basis[SpacedVolumeMetadata, np.complex128]:
        return self.get_fundamental_basis(cell)


@dataclass(frozen=True, kw_only=True)
class MomentumSimulationBasis(SimulationBasis):
    """The truncated basis for a simulation."""

    truncation: tuple[int, ...] | None = None
    """The number of states to truncate in each direction."""

    @override
    def with_resolution(self, resolution: tuple[int, ...]) -> MomentumSimulationBasis:
        return type(self)(shape=self.shape, resolution=resolution)

    @override
    def with_shape(self, shape: tuple[int, ...]) -> MomentumSimulationBasis:
        return type(self)(shape=shape, resolution=self.resolution)

    def with_truncation(self, truncation: tuple[int, ...]) -> MomentumSimulationBasis:
        """Create a new basis with a different truncation."""
        return type(self)(
            shape=self.shape, resolution=self.resolution, truncation=truncation
        )

    @override
    def get_simulation_basis(
        self, cell: SimulationCell
    ) -> Basis[SpacedVolumeMetadata, np.complex128]:
        fundamental = self.get_fundamental_metadata(cell)
        truncation = self.truncation or fundamental.shape

        return tuple_basis(
            tuple(
                basis.CroppedBasis(
                    s,
                    basis.TransformedBasis(basis.FundamentalBasis(m)),
                )
                for s, m in zip(truncation, fundamental.children, strict=False)
            ),
            fundamental.extra,
        )

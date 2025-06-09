from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from slate_core import Ctype, TupleBasis, TupleMetadata, basis
from slate_core.basis import AsUpcast
from slate_core.metadata import (
    AxisDirections,
    Domain,
    EvenlySpacedLengthMetadata,
    EvenlySpacedVolumeMetadata,
)
from slate_quantum import metadata

if TYPE_CHECKING:
    import numpy as np
    from slate_core.basis import Basis
    from slate_quantum.metadata import RepeatedVolumeMetadata
    from slate_quantum.operator import OperatorBasis


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

    def get_repeat_metadata(self, cell: SimulationCell) -> EvenlySpacedVolumeMetadata:
        """Get the metadata for the repeat cell of the simulation."""
        return TupleMetadata(
            tuple(
                EvenlySpacedLengthMetadata(r, domain=Domain(delta=delta))
                for r, delta in zip(self.resolution, cell.lengths, strict=False)
            ),
            cell.directions,
        )

    def get_repeat_basis(
        self, cell: SimulationCell
    ) -> Basis[EvenlySpacedVolumeMetadata, Ctype[np.complex128]]:
        """Get the basis for the repeat cell of the simulation."""
        return basis.from_metadata(self.get_repeat_metadata(cell)).upcast()

    def get_fundamental_metadata(self, cell: SimulationCell) -> RepeatedVolumeMetadata:
        """Get the metadata for the fundamental cell of the simulation."""
        return metadata.repeat_volume_metadata(
            self.get_repeat_metadata(cell), self.shape
        )

    def get_fundamental_basis(
        self, cell: SimulationCell
    ) -> Basis[EvenlySpacedVolumeMetadata]:
        """Get the fundamental basis for the simulation."""
        return basis.from_metadata(self.get_fundamental_metadata(cell)).upcast()

    def get_operator_basis(
        self, cell: SimulationCell
    ) -> OperatorBasis[EvenlySpacedVolumeMetadata]:
        """Get the basis for the simulation."""
        state_basis = self.get_simulation_basis(cell)
        return TupleBasis((state_basis, state_basis.dual_basis())).upcast()

    @abstractmethod
    def get_simulation_basis(
        self, cell: SimulationCell
    ) -> Basis[EvenlySpacedVolumeMetadata]:
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
    ) -> Basis[EvenlySpacedVolumeMetadata]:
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
    ) -> Basis[EvenlySpacedVolumeMetadata]:
        fundamental = self.get_fundamental_metadata(cell)
        truncation = self.truncation or fundamental.shape

        return AsUpcast(
            TupleBasis(
                tuple(
                    basis.CroppedBasis(
                        s, basis.TransformedBasis(basis.FundamentalBasis(m))
                    )
                    for s, m in zip(truncation, fundamental.children, strict=False)
                ),
                fundamental.extra,
            ),
            fundamental,
        )


@dataclass(frozen=True, kw_only=True)
class PositionSimulationBasis(SimulationBasis):
    """The truncated basis for a simulation."""

    truncation: tuple[int, ...] | None = None
    offset: tuple[int, ...] | None = None
    """The number of states to truncate in each direction."""

    @override
    def with_resolution(self, resolution: tuple[int, ...]) -> PositionSimulationBasis:
        return type(self)(shape=self.shape, resolution=resolution)

    @override
    def with_shape(self, shape: tuple[int, ...]) -> PositionSimulationBasis:
        return type(self)(shape=shape, resolution=self.resolution)

    def with_truncation(self, truncation: tuple[int, ...]) -> PositionSimulationBasis:
        """Create a new basis with a different truncation."""
        return type(self)(
            shape=self.shape, resolution=self.resolution, truncation=truncation
        )

    @override
    def get_simulation_basis(
        self, cell: SimulationCell
    ) -> Basis[EvenlySpacedVolumeMetadata]:
        fundamental = self.get_fundamental_metadata(cell)
        truncation = self.truncation or fundamental.shape
        offset = self.offset or tuple(0 for _ in fundamental.shape)
        children = fundamental.children

        return AsUpcast(
            TupleBasis(
                tuple(
                    basis.TruncatedBasis(
                        basis.Truncation(s, 1, o), basis.FundamentalBasis(m)
                    )
                    for s, o, m in zip(truncation, offset, children, strict=False)
                ),
                fundamental.extra,
            ),
            fundamental,
        )

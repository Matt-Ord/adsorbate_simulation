from __future__ import annotations

from typing import (
    Any,
)

import numpy as np

from slate.util import (
    get_position_in_sorted,
    slice_ignoring_axes,
)

from ._volume import (
    SpacedVolumeMetadata,
    fundamental_stacked_delta_k,
    fundamental_stacked_delta_x,
    fundamental_stacked_k_points,
    fundamental_stacked_x_points,
)


def _project_points_along_directions(
    points: tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
    directions: tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    projected_axes = np.zeros(np.shape(directions))
    for i, direction in enumerate(directions):
        projected = direction.copy()
        for j in range(i):
            projected -= projected_axes[j] * np.dot(projected_axes[j], projected)

        projected_axes[i] = projected / np.linalg.norm(projected)

    return np.tensordot(projected_axes, points, axes=(1, 0))


def _project_k_points_along_axes(
    points: tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
    metadata: SpacedVolumeMetadata,
    axes: tuple[int, ...],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Get the list of k points projected onto the plane including both axes."""
    directions = fundamental_stacked_delta_k(metadata)
    return _project_points_along_directions(
        points, tuple(directions[ax] for ax in axes)
    )


def get_fundamental_stacked_k_points_projected_along_axes(
    metadata: SpacedVolumeMetadata,
    axes: tuple[int, ...],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get the fundamental_k_points projected onto the plane including both axes."""
    points = fundamental_stacked_k_points(metadata)
    return _project_k_points_along_axes(points, metadata, axes)


def get_k_coordinates_in_axes(
    metadata: SpacedVolumeMetadata,
    axes: tuple[int, ...],
    idx: tuple[int, ...] | None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get the fundamental_k_points projected onto the plane including both axes."""
    idx = (
        tuple(0 for _ in range(len(metadata.children) - len(axes)))
        if idx is None
        else idx
    )
    points = get_fundamental_stacked_k_points_projected_along_axes(metadata, axes)
    slice_ = slice_ignoring_axes(idx, axes)
    return np.transpose(
        points.reshape(-1, *metadata.fundamental_shape)[:, *slice_],
        (0, *(1 + np.array(get_position_in_sorted(axes)))),
    )


def _project_x_points_along_axes(
    points: tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
    metadata: SpacedVolumeMetadata,
    axes: tuple[int, ...],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Get the list of k points projected onto the plane including both axes."""
    directions = fundamental_stacked_delta_x(metadata)
    return _project_points_along_directions(
        points, tuple(directions[ax] for ax in axes)
    )


def get_fundamental_stacked_x_points_projected_along_axes(
    metadata: SpacedVolumeMetadata,
    axes: tuple[int, ...],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get the fundamental_x_points projected onto the plane including both axes."""
    points = fundamental_stacked_x_points(metadata)
    return _project_x_points_along_axes(points, metadata, axes)


def get_x_coordinates_in_axes(
    metadata: SpacedVolumeMetadata,
    axes: tuple[int, ...],
    idx: tuple[int, ...] | None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get the fundamental_x_points projected onto the plane including both axes."""
    idx = (
        tuple(0 for _ in range(len(metadata.children) - len(axes)))
        if idx is None
        else idx
    )
    points = get_fundamental_stacked_x_points_projected_along_axes(metadata, axes)
    slice_ = slice_ignoring_axes(idx, axes)
    return np.transpose(
        points.reshape(-1, *metadata.fundamental_shape)[:, *slice_],
        (0, *(1 + np.array(get_position_in_sorted(axes)))),
    )

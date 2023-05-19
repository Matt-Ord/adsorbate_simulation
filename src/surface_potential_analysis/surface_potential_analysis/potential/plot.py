from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.basis.basis import BasisUtil
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfigUtil,
    calculate_cumulative_x_distances_along_path,
    get_fundamental_projected_x_points,
)
from surface_potential_analysis.util.plot import (
    animate_through_surface,
)
from surface_potential_analysis.util.util import (
    slice_along_axis,
)

from ._comparison_points import (
    get_100_comparison_points_x2,
    get_111_comparison_points_x2,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.animation import ArtistAnimation
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis._types import SingleFlatIndexLike
    from surface_potential_analysis.util.plot import Scale

    from .potential import Potential


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def plot_potential_1d(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    idx: tuple[int, int],
    axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the potential along the given axis.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    idx : tuple[int, int]
    axis : Literal[0, 1, 2, -1, -2, -3]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    util = BasisUtil(potential["basis"][axis])
    coordinates = np.linalg.norm(util.fundamental_x_points, axis=0)
    data_slice: list[slice | int] = [slice(None), slice(None), slice(None)]
    data_slice[1 if (axis % 3) == 0 else 0] = idx[0]
    data_slice[1 if (axis % 3) == 2 else 2] = idx[1]  # noqa: PLR2004
    data = potential["points"][tuple(data_slice)]

    (line,) = ax.plot(coordinates, data)
    ax.set_xlabel(f"x{(axis % 3)} axis")
    ax.set_ylabel("Energy /J")
    ax.set_yscale(scale)
    return fig, ax, line


def plot_potential_1d_comparison(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    comparison_points: Mapping[
        str, tuple[tuple[int, int], Literal[0, 1, 2, -1, -2, -3]]
    ],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot the potential in 1d at the given comparison points.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    comparison_points : Mapping[ str, tuple[tuple[int, int], Literal[0, 1, 2,
        map of axis label to ((idx), axis) to pass to plot_potential_1d
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    for label, (idx, axis) in comparison_points.items():
        (_, _, line) = plot_potential_1d(potential, idx, axis, ax=ax, scale=scale)
        line.set_label(label)
    ax.legend()
    return fig, ax


def plot_potential_1d_x2_comparison_111(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    offset: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot the potential along the x2 at the relevant 111 sites.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    offset : tuple[int, int]
        index of the fcc site
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    (s0, s1, _) = BasisConfigUtil(potential["basis"]).shape
    points = get_111_comparison_points_x2((s0, s1), offset)
    comparison_points = {k: (v, 2) for (k, v) in points.items()}
    return plot_potential_1d_comparison(
        potential, comparison_points, ax=ax, scale=scale  # type: ignore[arg-type]
    )


def plot_potential_1d_x2_comparison_100(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    offset: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot the potential along the x2 at the relevant 100 sites.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    offset : tuple[int, int]
        index of the fcc site
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    (s0, s1, _) = BasisConfigUtil(potential["basis"]).shape
    points = get_100_comparison_points_x2((s0, s1), offset)
    comparison_points = {k: (v, 2) for (k, v) in points.items()}
    return plot_potential_1d_comparison(
        potential, comparison_points, ax=ax, scale=scale  # type: ignore[arg-type]
    )


def plot_potential_2d_x(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    idx: SingleFlatIndexLike,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the potential in 2d, perpendicular to z_axis at idx along z_axis.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    idx : SingleFlatIndexLike
        index along z_axis
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis perpendicular to direction of plot
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    coordinates = get_fundamental_projected_x_points(potential["basis"], z_axis)[
        slice_along_axis(idx, (z_axis % 3) + 1)
    ]
    data = potential["points"][slice_along_axis(idx, z_axis)]

    mesh = ax.pcolormesh(*coordinates, data, shading="nearest")
    mesh.set_norm(scale)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"x{0 if (z_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"x{2 if (z_axis % 3) != 2 else 1} axis")  # noqa: PLR2004

    return fig, ax, mesh


def plot_potential_x0x1(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    x3_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the potential in 2d, perpendicular to x3 at x3_idx.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    x3_idx : SingleFlatIndexLike
        index along x3_axis
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    return plot_potential_2d_x(potential, x3_idx, 2, ax=ax, scale=scale)


def plot_potential_x1x2(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    x0_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the potential in 2d, perpendicular to x2 at x2_idx.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    x2_idx : SingleFlatIndexLike
        index along x2_axis
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    return plot_potential_2d_x(potential, x0_idx, 0, ax=ax, scale=scale)


def plot_potential_x2x0(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    x1_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the potential in 2d, perpendicular to x1 at x1_idx.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    x1_idx : SingleFlatIndexLike
        index along x1_axis
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    return plot_potential_2d_x(potential, x1_idx, 1, ax=ax, scale=scale)


def plot_potential_difference_2d_x(
    potential0: Potential[_L0Inv, _L1Inv, _L2Inv],
    potential1: Potential[_L0Inv, _L1Inv, _L2Inv],
    idx: SingleFlatIndexLike,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the difference between two potentials in 2d, perpendicular to z_axis.

    Parameters
    ----------
    potential0 : Potential[_L0Inv, _L1Inv, _L2Inv]
    potential1 : Potential[_L0Inv, _L1Inv, _L2Inv]
    idx : SingleFlatIndexLike
        index along z_axis
    z_axis : Literal[0, 1, 2,
        axis perpendicular to which to plot the data
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    potential: Potential[_L0Inv, _L1Inv, _L2Inv] = {
        "basis": potential0["basis"],
        "points": potential0["points"] - potential1["points"],
    }
    return plot_potential_2d_x(potential, idx, z_axis, ax=ax, scale=scale)


def animate_potential_3d_x(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate potential in the direction perpendicular to z_axis.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis to animate through
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    coordinates = get_fundamental_projected_x_points(potential["basis"], z_axis)

    fig, ax, ani = animate_through_surface(
        coordinates, potential["points"], z_axis, ax=ax, scale=scale, clim=clim
    )
    ax.set_title(f"Animation of the potential perpendicular to the x{z_axis % 3} axis")

    return fig, ax, ani


def animate_potential_x0x1(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the potential in the direction perpendicular to x0x1.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    ax : Axes | None, optional
        plot axes, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_potential_3d_x(potential, 2, ax=ax, scale=scale, clim=clim)


def animate_potential_x1x2(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the potential in the direction perpendicular to x0x1.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    ax : Axes | None, optional
        plot axes, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_potential_3d_x(potential, 0, ax=ax, scale=scale, clim=clim)


def animate_potential_x2x0(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the potential in the direction perpendicular to x0x1.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    ax : Axes | None, optional
        plot axes, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_potential_3d_x(potential, 1, ax=ax, scale=scale, clim=clim)


def animate_potential_difference_2d_x(
    potential0: Potential[_L0Inv, _L1Inv, _L2Inv],
    potential1: Potential[_L0Inv, _L1Inv, _L2Inv],
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Animate the absolute relative difference between two potentials in 2d, perpendicular to z_axis.

    Parameters
    ----------
    potential0 : Potential[_L0Inv, _L1Inv, _L2Inv]
    potential1 : Potential[_L0Inv, _L1Inv, _L2Inv]
    z_axis : Literal[0, 1, 2,
        axis perpendicular to which to plot the data
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    potential: Potential[_L0Inv, _L1Inv, _L2Inv] = {
        "basis": potential0["basis"],
        "points": np.abs(
            (potential0["points"] - potential1["points"]) / potential0["points"]
        ),
    }
    return animate_potential_3d_x(potential, z_axis, ax=ax, scale=scale, clim=clim)


def plot_potential_along_path(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the potential along the given path.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    path : np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
    wrap_distances : bool, optional
        should the coordinates be wrapped into the unit cell, by default False
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    data = potential["points"][*path]
    distances = calculate_cumulative_x_distances_along_path(
        potential["basis"], path, wrap_distances=wrap_distances
    )
    (line,) = ax.plot(distances, data)
    ax.set_yscale(scale)
    return fig, ax, line


def get_minimum_path(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    path: np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]],
    axis: Literal[0, 1, 2, -1, -2, -3] = -1,
) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]:
    """
    Find the minimum path, taking the smallest value along the given axis.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
        Potential to take the minimum of
    path : np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]]
        Path for the coordinates perpendicular to axis
    axis : Literal[0, 1, 2,-1,-2,-3] optional
        Axis over which to search for the minimum of potential, by default -1

    Returns
    -------
    np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        The complete path after finding the minimum at each coordinate
    """
    min_idx = np.argmin(potential["points"], axis=axis)
    min_idx_path = min_idx[*path]
    full_path = np.empty((3, path.shape[1]))
    full_path[1 if (axis % 3) == 0 else 0] = path[0]
    full_path[1 if (axis % 3) == 2 else 3] = path[1]  # noqa: PLR2004
    full_path[axis] = min_idx_path
    return full_path  # type: ignore[no-any-return]


def plot_potential_minimum_along_path(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    path: np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]],
    axis: Literal[0, 1, 2, -1, -2, -3] = -1,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the potentail along the path, taking the minimum in the axis direction.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    path : np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]]
        path perpendicular to axis
    axis : Literal[0, 1, 2, -1, -2, -3], optional
        axis to take the minimum along, by default -1
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    full_path = get_minimum_path(potential, path, axis)
    return plot_potential_along_path(potential, full_path, ax=ax, scale=scale)

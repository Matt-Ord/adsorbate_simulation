import math

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike, NDArray

from surface_potential_analysis.surface_config import get_surface_xy_points

from .energy_data import (
    EnergyGrid,
    EnergyPoints,
    add_back_symmetry_points,
    get_energy_grid_xy_points,
    get_energy_points_xy_locations,
)


def plot_energy_grid_points(
    grid: EnergyGrid, *, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = get_energy_grid_xy_points(grid)
    (line,) = ax.plot(points[:, 0], points[:, 1])
    line.set_marker("x")
    line.set_linestyle("")

    ax.set_aspect("equal", adjustable="box")

    return fig, ax, line


def plot_energy_in_z_direction(
    grid: EnergyGrid, xy_ind: tuple[int, int], *, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = np.array(grid["points"], dtype=float)
    (line,) = ax.plot(grid["z_points"], points[xy_ind[0], xy_ind[1]])

    return fig, ax, line


def get_111_locations(grid: EnergyGrid):
    points = np.array(grid["points"], dtype=float)
    return {
        "Top Site": (
            math.floor(2 * points.shape[0] / 3),
            math.floor(2 * points.shape[1] / 3),
        ),
        "Bridge Site": (
            math.floor(points.shape[0] / 6),
            math.floor(points.shape[1] / 6),
        ),
        "FCC Site": (0, 0),
        "HCP Site": (
            math.floor(points.shape[0] / 3),
            math.floor(points.shape[1] / 3),
        ),
    }


def get_100_locations(grid: EnergyGrid):

    points = np.array(grid["points"], dtype=float)
    return {
        "Top Site": (0, 0),
        "Bridge Site": (math.floor(points.shape[0] / 2), 0),
        "Hollow Site": (
            math.floor(points.shape[0] / 2),
            math.floor(points.shape[1] / 2),
        ),
    }


def plot_z_direction_energy_data(
    grid: EnergyGrid, locations: dict[str, tuple[int, int]], *, ax: Axes | None = None
) -> tuple[Figure, Axes, list[Line2D]]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    lines: list[Line2D] = []
    for (label, xy_ind) in locations.items():
        _, _, line = plot_energy_in_z_direction(grid, xy_ind, ax=ax)
        line.set_label(label)
        lines.append(line)

    ax.set_ylabel("Energy / J")
    ax.set_xlabel("relative z position /m")

    ax.legend()

    return fig, ax, lines


def plot_z_direction_energy_data_111(
    grid: EnergyGrid, *, ax: Axes | None = None
) -> tuple[Figure, Axes, tuple[Line2D, Line2D, Line2D, Line2D]]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    locations = get_111_locations(grid)
    fig, ax, lines = plot_z_direction_energy_data(grid, locations, ax=ax)
    return fig, ax, (lines[0], lines[1], lines[2], lines[3])


def plot_z_direction_energy_comparison_111(
    grid: EnergyGrid, otherData: EnergyGrid, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    plot_z_direction_energy_data_111(grid, ax=ax)
    _, _, (l1, l2, l3, l4) = plot_z_direction_energy_data_111(otherData, ax=ax)
    l1.set_linestyle("")
    l1.set_marker("x")
    l2.set_linestyle("")
    l2.set_marker("x")
    l3.set_linestyle("")
    l3.set_marker("x")
    l4.set_linestyle("")
    l4.set_marker("x")

    ax.legend()

    return fig, ax


def plot_z_direction_energy_data_100(
    grid: EnergyGrid, *, ax: Axes | None = None
) -> tuple[Figure, Axes, tuple[Line2D, Line2D, Line2D]]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    locations = get_100_locations(grid)
    fi, ax, lines = plot_z_direction_energy_data(grid, locations, ax=ax)

    return fig, ax, (lines[0], lines[1], lines[2])


def plot_z_direction_energy_comparison_100(
    grid: EnergyGrid, otherData: EnergyGrid, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    plot_z_direction_energy_data_100(grid, ax=ax)
    _, _, (l1, l2, l3) = plot_z_direction_energy_data_100(otherData, ax=ax)
    l1.set_linestyle("")
    l1.set_marker("x")
    l2.set_linestyle("")
    l2.set_marker("x")
    l3.set_linestyle("")
    l3.set_marker("x")

    return fig, ax


# Note assumes orthogonal
def plot_x_direction_energy_data(data: EnergyGrid) -> None:
    fig, ax = plt.subplots()

    points = np.array(add_back_symmetry_points(data["points"]))
    heights = np.linspace(0, data["delta_x1"][0], points.shape[0])
    middle_x_index = math.floor(points.shape[0] / 2)
    middle_y_index = math.floor(points.shape[1] / 2)
    top_equilibrium = np.argmin(points[0, 0])
    hollow_equilibrium = np.argmin(points[middle_x_index, middle_y_index])

    hollow_eq_energy = points[:, middle_y_index, hollow_equilibrium]
    top_eq_energy = points[:, middle_y_index, top_equilibrium]
    hollow_max_energy = points[:, middle_y_index, 0]

    ax.plot(heights, top_eq_energy, label="Near Top Equilibrium")
    ax.plot(heights, hollow_eq_energy, label="Near Hollow Equilibrium")
    ax.plot(heights, hollow_max_energy, label="Near Hollow Maximum")

    ax.set_title("Plot of energy in the x direction")
    ax.set_ylabel("Energy / eV")
    ax.set_xlabel("relative z position /m")
    ax.legend()

    fig.tight_layout()
    fig.show()
    fig.savefig("temp.png")


def plot_xz_plane_energy_copper_100(data: EnergyGrid) -> Figure:
    fig, axs = plt.subplots(nrows=2, ncols=3)

    points = np.array(add_back_symmetry_points(data["points"]))
    x_points = np.linspace(0, data["delta_x0"][0], points.shape[0])
    y_points = np.linspace(0, data["delta_x1"][1], points.shape[0])
    z_points = np.array(data["z_points"])
    middle_x_index = math.floor(points.shape[0] / 2)
    middle_y_index = math.floor(points.shape[1] / 2)
    max_potential = 1e-18

    bridge_energies = np.clip(points[::, 0, ::-1].transpose(), 0, max_potential)
    hollow_energies = np.clip(
        points[::, middle_x_index, ::-1].transpose(), 0, max_potential
    )
    top_hollow_energies = np.clip(points.diagonal()[::-1], 0, max_potential)

    extent = [x_points[0], x_points[-1], z_points[0], z_points[-1]]
    axs[0][0].imshow(bridge_energies, extent=extent)
    axs[0][2].imshow(hollow_energies, extent=extent)
    extent = [
        np.sqrt(2) * x_points[0],
        np.sqrt(2) * x_points[-1],
        data["z_points"][0],
        data["z_points"][-1],
    ]
    axs[0][1].imshow(top_hollow_energies, extent=extent)

    extent = [x_points[0], x_points[-1], y_points[0], y_points[-1]]
    bottom_energies = np.clip(points[::, ::, 0], 0, max_potential)
    axs[1][0].imshow(bottom_energies, extent=extent)
    equilibrium_z = np.argmin(points[middle_x_index, middle_y_index])
    equilibrium_energies = np.clip(points[::, ::, equilibrium_z], 0, max_potential)
    axs[1][2].imshow(equilibrium_energies, extent=extent)

    axs[0][1].sharey(axs[0][0])
    axs[0][2].sharey(axs[0][0])
    axs[1][0].sharex(axs[0][0])
    axs[1][2].sharex(axs[0][2])

    axs[0][0].set_xlabel("x Position")
    axs[0][0].set_ylabel("z position /m")

    axs[0][0].set_title("Top-Bridge Site")
    axs[0][1].set_title("Top-Hollow Site")
    axs[0][2].set_title("Bridge-Hollow Site")
    axs[1][0].set_title("Bottom Energies")
    axs[1][2].set_title("Equilibrium Energies")

    fig.suptitle("Plot of energy through several planes perpendicular to xy")
    fig.tight_layout()
    return fig


def plot_energy_point_z(
    energy_points: EnergyPoints, xy: tuple[float, float], ax: Axes | None = None
):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    idx = np.argwhere(
        np.logical_and(
            np.array(energy_points["x_points"]) == xy[0],
            np.array(energy_points["y_points"]) == xy[1],
        )
    )
    points = np.array(energy_points["points"])[idx]
    z_points = np.array(energy_points["z_points"])[idx]

    (line,) = ax.plot(z_points, points)
    ax.set_xlabel("z")
    ax.set_ylabel("Energy")
    return fig, ax, line


def plot_all_energy_points_z(energy_points: EnergyPoints, ax: Axes | None = None):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = get_energy_points_xy_locations(energy_points)
    for (x, y) in points:
        _, _, line = plot_energy_point_z(energy_points, (x, y), ax)
        line.set_label(f"{x:.2}, {y:.2}")

    ax.set_title("Plot of Energy against z for each (x,y) point")
    return fig, ax


def plot_energy_points_location(energy_points: EnergyPoints, ax: Axes | None = None):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = get_energy_points_xy_locations(energy_points)
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]
    (line,) = ax.plot(x_points, y_points)
    line.set_marker("x")
    line.set_linestyle("")
    ax.set_aspect("equal", adjustable="box")

    return fig, ax, line


def plot_energy_grid_in_xy(
    grid: EnergyGrid, z_ind: int, *, ax: Axes | None = None
) -> tuple[Figure, Axes, QuadMesh]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = np.array(grid["points"])
    coordinates = get_energy_grid_xy_points(grid).reshape(
        points.shape[0], points.shape[1], 2
    )
    mesh = ax.pcolormesh(
        coordinates[:, :, 0],
        coordinates[:, :, 1],
        points[:, :, z_ind],
        shading="nearest",
    )
    ax.set_aspect("equal", adjustable="box")
    return (fig, ax, mesh)


def animate_energy_grid_3D_in_xy(
    grid: EnergyGrid, *, ax: Axes | None = None, clim_max: float | None = None
) -> tuple[Figure, Axes, matplotlib.animation.ArtistAnimation]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    points = np.array(grid["points"])
    clim_max = clim_max if clim_max is not None else np.max(points)

    _, _, mesh = plot_energy_grid_in_xy(grid, 0, ax=ax)
    mesh.set_norm("symlog")  # type: ignore
    mesh.set_clim(0, clim_max)

    frames: list[list[QuadMesh]] = []
    for z_ind in range(points.shape[2]):

        _, _, mesh = plot_energy_grid_in_xy(grid, z_ind, ax=ax)
        mesh.set_norm("symlog")  # type: ignore
        mesh.set_clim(0, clim_max)

        frames.append([mesh])

    ani = matplotlib.animation.ArtistAnimation(fig, frames)

    ax.set_xlabel("X direction")
    ax.set_ylabel("Y direction")
    ax.set_aspect("equal", adjustable="box")

    fig.colorbar(mesh, ax=ax, format="%4.1e")

    return fig, ax, ani


def plot_energy_grid_in_x1z(
    grid: EnergyGrid, x2_ind: int, *, ax: Axes | None = None
) -> tuple[Figure, Axes, QuadMesh]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = np.array(grid["points"])
    x1_points = np.linspace(0, np.linalg.norm(grid["delta_x1"]), points.shape[0])
    z_points = np.array(grid["z_points"])
    x1v, zv = np.meshgrid(x1_points, z_points, indexing="ij")
    mesh = ax.pcolormesh(
        x1v,
        zv,
        points[:, x2_ind, :],
        shading="nearest",
    )
    ax.set_aspect("equal", adjustable="box")
    return (fig, ax, mesh)


def animate_energy_grid_3D_in_x1z(
    grid: EnergyGrid, *, ax: Axes | None = None, clim_max: float | None = None
) -> tuple[Figure, Axes, matplotlib.animation.ArtistAnimation]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    points = np.array(grid["points"])
    clim_max = clim_max if clim_max is not None else np.max(points)

    _, _, mesh = plot_energy_grid_in_x1z(grid, 0, ax=ax)
    mesh.set_clim(0, clim_max)
    mesh.set_norm("symlog")  # type: ignore

    frames: list[list[QuadMesh]] = []
    for x2_ind in range(points.shape[1]):

        _, _, mesh = plot_energy_grid_in_x1z(grid, x2_ind, ax=ax)
        mesh.set_clim(0, clim_max)
        mesh.set_norm("symlog")  # type: ignore
        frames.append([mesh])

    ani = matplotlib.animation.ArtistAnimation(fig, frames)

    ax.set_xlabel("X1 direction")
    ax.set_ylabel("Z direction")
    ax.set_aspect("equal", adjustable="box")

    fig.colorbar(mesh, ax=ax, format="%4.1e")

    return fig, ax, ani


def compare_energy_grid_to_all_raw_points(
    raw_points: EnergyPoints, grid: EnergyGrid, *, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = get_energy_points_xy_locations(raw_points)
    xy_points = get_energy_grid_xy_points(grid)
    x_points = xy_points[:, 0]
    y_points = xy_points[:, 1]
    grid_points = np.reshape(grid["points"], (-1, np.shape(grid["points"])[2]))

    cols = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    for (i, (x, y)) in enumerate(points):
        _, _, line = plot_energy_point_z(raw_points, (x, y), ax)
        line.set_color(cols[i])
        line.set_linestyle("")
        line.set_marker("x")

        ixy = (np.square(x_points - x) + np.square(y_points - y)).argmin()

        (line,) = ax.plot(grid["z_points"], grid_points[ixy, :])
        line.set_label(f"{x:.2}, {y:.2}")
        line.set_color(cols[i])

    return fig, ax


def plot_energy_grid_locations(
    grid: EnergyGrid, *, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = get_energy_grid_xy_points(grid)
    (line,) = ax.plot(points[:, 0], points[:, 1])
    line.set_linestyle("")
    line.set_marker("x")
    return fig, ax, line


def plot_energy_point_locations_on_grid(
    raw_points: EnergyPoints, grid: EnergyGrid, *, ax: Axes | None = None
) -> tuple[Figure, Axes, matplotlib.animation.ArtistAnimation]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = get_energy_points_xy_locations(raw_points)
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]
    (line,) = ax.plot(x_points, y_points)
    line.set_marker("x")
    line.set_linestyle("")

    z_points = np.unique(raw_points["z_points"])

    grid_z_points = np.array(grid["z_points"])
    clim = (np.min(grid["points"]), np.max(grid["points"]))

    frames: list[tuple[QuadMesh, Line2D]] = []

    for z in z_points:

        iz = (np.abs(grid_z_points - z)).argmin()

        _, _, mesh = plot_energy_grid_in_xy(grid, iz, ax=ax)
        mesh.set_clim(*clim)
        frames.append((mesh, line))

    ani = matplotlib.animation.ArtistAnimation(fig, frames, interval=500)

    ax.set_xlabel("X line")
    ax.set_ylabel("Y line")
    ax.set_aspect("equal", adjustable="box")

    return fig, ax, ani


def calculate_cumulative_distances_along_path(path: ArrayLike, coordinates: NDArray):
    """
    Get a list of cumulative distances along a path (list[coordinate]) given a grid of coordinates

    Parameters
    ----------
    path : NDArray
        Path for which to calculate the distance along (as a list of coordinates)
    coordinates : NDArray
        Coordinate grid, with the same simension as the coordinates given in the path
    """
    coordinate_shape = np.shape(coordinates)[0:-1]
    path_index = np.ravel_multi_index(
        np.moveaxis(path, -1, 0), coordinate_shape  # type:ignore
    )
    path_coordinates = coordinates.reshape(-1, coordinates.shape[-1])[path_index]

    distances = np.linalg.norm(path_coordinates[:-1] - path_coordinates[1:], axis=-1)
    cum_distances = np.cumsum(distances)
    # Add back initial distance
    return np.insert(cum_distances, 0, 0)


def plot_potential_minimum_along_path(
    grid: EnergyGrid, path: list[tuple[int, int]], *, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the minimum of the potential along the given path.

    Note this path is not guaranteed to be continuous in the z direction

    Parameters
    ----------
    grid : EnergyGrid
    path : list[tuple[int, int]]
    ax   : Axes | None, optional


    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    path_arr = np.array(path)
    points = np.min(grid["points"], axis=-1)[path_arr[:, 0], path_arr[:, 1]]

    coordinates = get_surface_xy_points(
        grid, (np.shape(grid["points"])[0], np.shape(grid["points"])[1])
    )
    distances = calculate_cumulative_distances_along_path(path, coordinates)

    (line,) = ax.plot(distances, points)
    ax.set_xlabel("Distance / M")
    ax.set_ylabel("Energy / J")

    return fig, ax, line

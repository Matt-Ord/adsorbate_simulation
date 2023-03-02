from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from nickel_111.s2_hamiltonian import generate_hamiltonian_john
from nickel_111.s4_wavepacket import (
    get_brillouin_points_nickel_111,
    get_irreducible_config_nickel_111_supercell,
)
from surface_potential_analysis.eigenstate import EigenstateConfigUtil
from surface_potential_analysis.eigenstate_plot import (
    animate_eigenstate_3D_in_xy,
    plot_eigenstate_in_xy,
    plot_eigenstate_in_xz,
    plot_eigenstate_in_yz,
)
from surface_potential_analysis.energy_eigenstate import (
    EnergyEigenstates,
    filter_eigenstates_band,
    get_eigenstate_list,
    load_energy_eigenstates,
    load_energy_eigenstates_legacy,
    normalize_eigenstate_phase,
)
from surface_potential_analysis.energy_eigenstates_plot import plot_eigenstate_positions
from surface_potential_analysis.wavepacket_grid import (
    calculate_wavepacket_grid,
    calculate_wavepacket_grid_fourier,
    load_wavepacket_grid_legacy,
    save_wavepacket_grid,
)
from surface_potential_analysis.wavepacket_grid_plot import (
    animate_wavepacket_grid_3D_in_x1z,
    animate_wavepacket_grid_3D_in_xy,
)

from .surface_data import get_data_path, save_figure


def plot_wavepacket_points():
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)

    fig, _, _ = plot_eigenstate_positions(eigenstates)
    fig.show()

    eigenstate_list = get_eigenstate_list(filter_eigenstates_band(eigenstates))
    fig, _, _ = animate_eigenstate_3D_in_xy(
        eigenstates["eigenstate_config"], eigenstate_list[-1]
    )
    fig.show()
    input()


def select_single_k_eigenstates(
    eigenstates: EnergyEigenstates, kx: float, ky: float
) -> EnergyEigenstates:
    point_filter = np.logical_and(
        np.array(eigenstates["kx_points"]) == kx,
        np.array(eigenstates["ky_points"]) == ky,
    )
    out: EnergyEigenstates = {
        "kx_points": np.array(eigenstates["kx_points"])[point_filter].tolist(),
        "ky_points": np.array(eigenstates["ky_points"])[point_filter].tolist(),
        "eigenstate_config": eigenstates["eigenstate_config"],
        "eigenvalues": np.array(eigenstates["eigenvalues"])[point_filter].tolist(),
        "eigenvectors": np.array(eigenstates["eigenvectors"])[point_filter].tolist(),
    }
    return out


def plot_wavepacket_points_in_yz_from_list():
    path = get_data_path("eigenstates_grid_2.json")
    eigenstates = select_single_k_eigenstates(
        load_energy_eigenstates_legacy(path), 0, 0
    )
    fig, _, _ = plot_eigenstate_positions(eigenstates)

    fig.show()

    eigenstate_list = get_eigenstate_list(eigenstates)
    config = eigenstates["eigenstate_config"]

    fig, _, _anim1 = plot_eigenstate_in_yz(config, eigenstate_list[0], measure="real")
    fig.show()

    fig, _, _anim2 = plot_eigenstate_in_yz(config, eigenstate_list[1], measure="real")
    fig.show()

    fig, _, _anim3 = plot_eigenstate_in_yz(config, eigenstate_list[2], measure="real")
    fig.show()

    fig, _, _anim4 = plot_eigenstate_in_yz(config, eigenstate_list[3], measure="real")
    fig.show()
    input()


def plot_wavepacket_points_in_xz_from_list():
    path = get_data_path("eigenstates_grid_2.json")
    eigenstates = select_single_k_eigenstates(
        load_energy_eigenstates_legacy(path), 0, 0
    )
    fig, _, _ = plot_eigenstate_positions(eigenstates)

    fig.show()

    eigenstate_list = get_eigenstate_list(eigenstates)
    config = eigenstates["eigenstate_config"]

    fig, _, _anim1 = plot_eigenstate_in_xz(config, eigenstate_list[0], measure="real")
    fig.show()

    fig, _, _anim2 = plot_eigenstate_in_xz(config, eigenstate_list[1], measure="real")
    fig.show()

    fig, _, _anim3 = plot_eigenstate_in_xz(config, eigenstate_list[2], measure="real")
    fig.show()

    fig, _, _anim4 = plot_eigenstate_in_xz(config, eigenstate_list[3], measure="real")
    fig.show()
    input()


def plot_wavepacket_points_in_xy_from_list():
    path = get_data_path("eigenstates_grid_2.json")
    eigenstates = select_single_k_eigenstates(
        load_energy_eigenstates_legacy(path), 0, 0
    )
    fig, _, _ = plot_eigenstate_positions(eigenstates)

    fig.show()

    eigenstate_list = get_eigenstate_list(eigenstates)
    print([np.sum(np.square(np.abs(j["eigenvector"]))) for j in eigenstate_list])
    config = eigenstates["eigenstate_config"]

    fig, _, _anim1 = plot_eigenstate_in_xy(config, eigenstate_list[0], measure="real")
    fig.show()

    fig, _, _anim2 = plot_eigenstate_in_xy(config, eigenstate_list[1], measure="real")
    fig.show()

    fig, _, _anim3 = plot_eigenstate_in_xy(config, eigenstate_list[2], measure="real")
    fig.show()

    fig, _, _anim4 = plot_eigenstate_in_xy(config, eigenstate_list[3], measure="real")
    fig.show()
    input()


def plot_energy_of_first_bands():
    path = get_data_path("eigenstates_grid_2.json")
    eigenstates = load_energy_eigenstates_legacy(path)

    origin_k_eigenstates = select_single_k_eigenstates(eigenstates, 0, 0)
    fig, ax = plt.subplots()

    (line,) = ax.plot(origin_k_eigenstates["eigenvalues"])
    line.set_marker("x")
    line.set_linestyle("")

    ax.set_title("Plot of the energy of the first 4 bands in the super-lattuice")

    fig.show()
    save_figure(fig, "nickel_super_lattice_energy_4_band.png")
    input()


def calculate_wavepacket_grid_nickel_111(eigenstates: EnergyEigenstates):
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])

    return calculate_wavepacket_grid(
        eigenstates,
        util.delta_x0,
        util.delta_x1,
        np.linspace(-util.characteristic_z * 2, util.characteristic_z * 2, 11).tolist(),
        shape=(37, 25),
        offset=(0.0, 0.0),
    )


def get_value_at_point(
    eigenstates: EnergyEigenstates, point: Tuple[float, float, float]
):
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    return sum(
        util.calculate_wavefunction_fast(e, [point])[0]
        for e in get_eigenstate_list(eigenstates)
    )


def test_single_k_wavepacket():
    """
    Figuring out how to produce a 'localized' wavepacket from a single k-point
    ie only one site in the super cell
    """

    path = get_data_path("eigenstates_grid_2.json")
    eigenstates = load_energy_eigenstates_legacy(path)

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])

    # Note we don't need to worry about 'repeats'
    # Ie at (1.0 * util.delta_x / 3, util.delta_y, 0),
    origins = [
        (0, 1.0 * util.delta_x1[1] / 3, 0),
        (0, 2.0 * util.delta_x1[1] / 3, 0),
        (util.delta_x0[0] / 2, 0.5 * util.delta_x1[1] / 3, 0),
        (util.delta_x0[0] / 2, 2.5 * util.delta_x1[1] / 3, 0),
    ]

    lowest_band = select_single_k_eigenstates(
        eigenstates, eigenstates["kx_points"][0], eigenstates["ky_points"][0]
    )
    # lowest_band["eigenvalues"] = lowest_band["eigenvalues"][0:2]
    # lowest_band["kx_points"] = lowest_band["kx_points"][0:2]
    # lowest_band["ky_points"] = lowest_band["ky_points"][0:2]
    # lowest_band["eigenvectors"] = lowest_band["eigenvectors"][0:2]

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])

    path = get_data_path("single_band_wavepacket_1.json")
    normalized1 = normalize_eigenstate_phase(lowest_band, origins[0])
    grid1 = calculate_wavepacket_grid_nickel_111(normalized1)
    save_wavepacket_grid(grid1, path)
    grid1 = load_wavepacket_grid_legacy(path)

    path = get_data_path("single_band_wavepacket_2.json")
    # normalized2 = normalize_eigenstate_phase(lowest_band, origins[1])
    # grid2 = calculate_wavepacket_grid_nickel_111(normalized2)
    # save_wavepacket_grid(grid2, path)
    grid2 = load_wavepacket_grid_legacy(path)

    path = get_data_path("single_band_wavepacket_3.json")
    # normalized3 = normalize_eigenstate_phase(lowest_band, origins[2])
    # grid3 = calculate_wavepacket_grid_nickel_111(normalized3)
    # save_wavepacket_grid(grid3, path)
    grid3 = load_wavepacket_grid_legacy(path)

    fig, ax, _anim1 = animate_wavepacket_grid_3D_in_xy(
        grid1, norm="linear", measure="real"
    )
    fig.show()

    fig, ax, _anim2 = animate_wavepacket_grid_3D_in_x1z(
        grid1, norm="linear", measure="real"
    )
    fig.show()

    fig, ax, _anim3 = animate_wavepacket_grid_3D_in_xy(
        grid2, norm="linear", measure="real"
    )
    fig.show()

    fig, ax, _anim4 = animate_wavepacket_grid_3D_in_xy(
        grid3, norm="linear", measure="real"
    )
    fig.show()

    print(get_value_at_point(normalized1, origins[0]))
    print(get_value_at_point(normalized1, origins[1]))
    print(get_value_at_point(normalized1, origins[2]))
    print(get_value_at_point(normalized1, origins[3]))
    print(eigenstates["kx_points"][0], eigenstates["ky_points"][0])
    input()


def plot_wavepacket_small():
    path = get_data_path("eigenstates_wavepacket_0_small.json")
    grid = load_wavepacket_grid_legacy(path)
    fig, ax, _anim0 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="real"
    )
    fig.show()

    path = get_data_path("eigenstates_wavepacket_1_small.json")
    grid = load_wavepacket_grid_legacy(path)
    fig, ax, _anim1 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="real"
    )
    fig.show()

    path = get_data_path("eigenstates_grid_2.json")
    eigenstates = load_energy_eigenstates_legacy(path)
    eigenstates["eigenstate_config"] = get_irreducible_config_nickel_111_supercell(
        eigenstates["eigenstate_config"]
    )

    fig, _, _ = plot_eigenstate_positions(eigenstates)
    fig.show()

    input()


def plot_wavepacket_points_john():

    a = np.array(
        [
            [+0, +2],
            [-1, +2],
            [+1, +1],
            [-2, +2],
            [+0, +1],
            [+2, +0],
            [-1, +1],
            [+1, +0],
            [-2, +1],
            [+0, +0],
            [+2, -1],
            [-1, +0],
            [+1, -1],
            [-2, +0],
            [+0, -1],
            [+2, -2],
            [-1, -1],
            [+1, -2],
            [+0, -2],
        ]
    )
    # = 2 x 2pi / delta_y
    G1 = 2.9419 * 10**10
    nqdim = 4

    qx_points = G1 * ((np.sqrt(3) / 2) * a[:, 0] + 0 * a[:, 1]) / nqdim
    qy_points = G1 * ((1 / 2) * a[:, 0] + a[:, 1]) / nqdim

    fig, ax = plt.subplots()
    (line,) = ax.plot(qx_points, qy_points)
    line.set_linestyle("")
    line.set_marker("x")
    ax.set_title("Plot of points as chosen by John")
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel("ky")
    ax.set_xlabel("kx")
    fig.show()
    input()
    save_figure(fig, "john_wavepacket_points.png")


def plot_wavepacket_points_me():
    hamiltonian = generate_hamiltonian_john()
    points = get_brillouin_points_nickel_111(hamiltonian._config, size=(2, 2))
    fig, ax = plt.subplots()
    (line,) = ax.plot(points[:, 0], points[:, 1])
    line.set_linestyle("")
    line.set_marker("x")
    ax.set_title("Plot of points as chosen by Me")
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel("ky")
    ax.set_xlabel("kx")
    fig.show()
    input()
    save_figure(fig, "my_wavepacket_points.png")


def plot_wavepacket_grid_all_equal():
    """
    Does the imaginary oscillation in the imaginary part of the wavefunction happen
    if we choose a constant bloch wavefunction for all k
    """
    path = get_data_path(f"eigenstates_grid_{0}.json")
    eigenstates = load_energy_eigenstates(path)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    eigenvector = np.zeros(np.prod(util.resolution))
    eigenvector[util.get_index(0, 0, 0)] = 1
    eigenstates["eigenvectors"] = [
        eigenvector.tolist() for _ in eigenstates["eigenvectors"]
    ]

    z_points = [0]
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points, x0_lim=(0, 10), x1_lim=(0, 10)
    )

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(grid, norm="symlog")
    fig.show()

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="real"
    )
    fig.show()

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="imag"
    )
    fig.show()
    input()


def plot_wavepacket_grid():

    path = get_data_path(f"eigenstates_grid_{0}.json")
    eigenstates = load_energy_eigenstates(path)
    eigenstates = normalize_eigenstate_phase(eigenstates, (0, 0, 0))

    z_points = [0]
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points, x0_lim=(0, 10), x1_lim=(0, 10)
    )

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(grid, norm="symlog")
    fig.show()

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="real"
    )
    fig.show()

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="imag"
    )
    fig.show()
    input()

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    path = get_data_path(f"eigenstates_grid_{1}.json")
    eigenstates = load_energy_eigenstates(path)
    eigenstates = normalize_eigenstate_phase(
        eigenstates,
        (
            (util.delta_x0[0] + util.delta_x1[0]) / 3,
            (util.delta_x0[1] + util.delta_x1[1]) / 3,
            0,
        ),
    )

    z_points = [0]
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points, x0_lim=(0, 10), x1_lim=(0, 10)
    )

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(grid, norm="symlog")
    fig.show()

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="real"
    )
    fig.show()

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="imag"
    )
    fig.show()
    input()
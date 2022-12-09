import datetime
import math
from functools import cache, cached_property, wraps
from typing import Any, Callable, Iterable, Tuple, TypeVar

import numpy as np
import scipy.special
from numpy.typing import ArrayLike, NDArray
from scipy.constants import hbar

import hamiltonian_diag
from energy_data import EnergyEigenstates, EnergyInterpolation
from sho_config import SHOConfig

F = TypeVar("F", bound=Callable)


def timed(f: F) -> F:
    @wraps(f)
    def wrap(*args, **kw):
        ts = datetime.datetime.now()
        result = f(*args, **kw)
        te = datetime.datetime.now()
        print(f"func: {f.__name__} took: {(te - ts).total_seconds()} sec")
        return result

    return wrap  # type: ignore


def calculate_sho_wavefunction(z_points, sho_omega, mass, n) -> NDArray:
    norm = (sho_omega * mass / hbar) ** 0.5
    normalized_z = z_points * norm

    prefactor = np.sqrt(norm / (2**n * math.factorial(n) * np.sqrt(np.pi)))
    hermite = scipy.special.eval_hermite(n, normalized_z)
    exponential = np.exp(-np.square(normalized_z) / 2)
    return prefactor * hermite * exponential


class SurfaceHamiltonian:

    _potential: EnergyInterpolation

    _config: SHOConfig

    _resolution: Tuple[int, int, int]

    def __init__(
        self,
        resolution: Tuple[int, int, int],
        potential: EnergyInterpolation,
        config: SHOConfig,
    ) -> None:

        self._potential = potential
        self._config = config
        self._resolution = resolution

        if (2 * self.Nkx) > self.Nx:
            print("Warning: max(ndkx) > Nx, some over sampling will occur")
        if (2 * self.Nky) > self.Ny:
            print("Warning: max(ndky) > Ny, some over sampling will occur")

    @property
    def points(self):
        return np.array(self._potential["points"])

    @property
    def mass(self):
        return self._config["mass"]

    @property
    def sho_omega(self):
        return self._config["sho_omega"]

    @property
    def z_offset(self):
        return self._config["z_offset"]

    @property
    def delta_x(self) -> float:
        return self._potential["delta_x"]

    @cached_property
    def dkx(self) -> float:
        return 2 * np.pi / self.delta_x

    @property
    def Nx(self) -> int:
        return self.points.shape[0]

    @property
    def Nkx(self) -> int:
        return 2 * self._resolution[0] + 1

    @property
    def x_points(self):
        """
        Calculate the lattice coordinates in the x direction

        Note: We don't store the 'nth' pixel
        """
        return np.linspace(0, self.delta_x, self.Nx, endpoint=False)

    @property
    def nkx_points(self):
        return np.arange(-self._resolution[0], self._resolution[0] + 1, dtype=int)

    @property
    def delta_y(self) -> float:
        return self._potential["delta_y"]

    @cached_property
    def dky(self) -> float:
        return 2 * np.pi / (self.delta_y)

    @property
    def Ny(self) -> int:
        return self.points.shape[1]

    @property
    def Nky(self) -> int:
        return 2 * self._resolution[1] + 1

    @property
    def y_points(self):
        """
        Calculate the lattice coordinates in the y direction

        Note: We don't store the 'nth' pixel
        """
        return np.linspace(0, self.delta_y, self.Ny, endpoint=False)

    @property
    def nky_points(self):
        return np.arange(-self._resolution[1], self._resolution[1] + 1, dtype=int)

    @property
    def dz(self) -> float:
        return self._potential["dz"]

    @property
    def Nz(self) -> int:
        return self.points.shape[2]

    @property
    def Nkz(self) -> int:
        return self._resolution[2]

    @property
    def z_points(self):
        z_start = self.z_offset
        z_end = self.z_offset + (self.Nz - 1) * self.dz
        return np.linspace(z_start, z_end, self.Nz)

    @property
    def nz_points(self):
        return np.arange(self.Nkz, dtype=int)

    @cached_property
    def coordinates(self) -> NDArray:
        xt, yt, zt = np.meshgrid(
            self.nkx_points,
            self.nky_points,
            self.nz_points,
            indexing="ij",
        )
        return np.array([xt.ravel(), yt.ravel(), zt.ravel()]).T

    @cache
    def hamiltonian(self, kx: float, ky: float) -> NDArray:
        diagonal_energies = np.diag(self._calculate_diagonal_energy(kx, ky))
        other_energies = self._calculate_off_diagonal_energies_fast()

        energies = diagonal_energies + other_energies
        if not np.allclose(energies, energies.conjugate().T):
            raise AssertionError("hamiltonian is not hermitian")
        return energies

    def eigenvalues(self, kx: float, ky: float) -> NDArray:
        w, _ = self._calculate_eigenvalues(kx, ky)
        return w

    def eigenvectors(self, kx: float, ky: float) -> NDArray:
        """
        Returns the eigenvalues as a list of vectors,
        ie v[i] is the eigenvector associated to the eigenvalue w[i]
        """
        _, v = self._calculate_eigenvalues(kx, ky)
        return v

    @cache
    @timed
    def _calculate_eigenvalues(self, kx: float, ky: float) -> Tuple[NDArray, NDArray]:
        print(kx, ky)
        w, v = np.linalg.eigh(self.hamiltonian(kx, ky))
        return (w, v.transpose())

    @timed
    def _calculate_diagonal_energy(self, kx: float, ky: float) -> NDArray[Any]:
        kx_coords, ky_coords, nz_coords = self.coordinates.T

        x_energy = (hbar * (self.dkx * kx_coords + kx)) ** 2 / (2 * self.mass)
        y_energy = (hbar * (self.dky * ky_coords + ky)) ** 2 / (2 * self.mass)
        z_energy = (hbar * self.sho_omega) * (nz_coords + 0.5)
        return x_energy + y_energy + z_energy

    def get_index(self, nkx: int, nky: int, nz: int) -> int:
        ikx = (nkx + self._resolution[0]) * self.Nky * self.Nkz
        iky = (nky + self._resolution[1]) * self.Nkz
        return ikx + iky + nz

    @cache
    def get_sho_potential(self) -> NDArray:
        return 0.5 * self.mass * self.sho_omega**2 * np.square(self.z_points)

    @cache
    def get_sho_subtracted_points(self) -> NDArray:
        return np.subtract(self.points, self.get_sho_potential())

    @cache
    def get_ft_potential(self) -> NDArray:
        subtracted_potential = self.get_sho_subtracted_points()
        fft_potential = np.fft.ifft2(subtracted_potential, axes=(0, 1))

        if not np.all(np.isreal(np.real_if_close(fft_potential))):
            raise AssertionError("FFT was not real!")
        return np.real_if_close(fft_potential)

    @cache
    def get_ft_potential_fast(self) -> NDArray:
        subtracted_potential = self.get_sho_subtracted_points()
        fft_potential = np.fft.ifft2(subtracted_potential, axes=(0, 1))
        return np.real_if_close(fft_potential)

    @cache
    def _calculate_sho_wavefunction_points(self, n: int) -> NDArray:
        """Generates the nth SHO wavefunction using the current config"""
        return calculate_sho_wavefunction(self.z_points, self.sho_omega, self.mass, n)

    @cache
    def _calculate_off_diagonal_entry(self, nz1, nz2, ndkx, ndky) -> float:
        """Calculates the off diagonal energy using the 'folded' points ndkx, ndky"""
        ft_pot_points = self.get_ft_potential()[ndkx, ndky]
        hermite1 = self._calculate_sho_wavefunction_points(nz1)
        hermite2 = self._calculate_sho_wavefunction_points(nz2)

        fourier_transform = np.sum(hermite1 * hermite2 * ft_pot_points)

        return self.dz * fourier_transform

    @cache
    @timed
    def _calculate_off_diagonal_energies_fast(self) -> NDArray:

        return np.array(
            hamiltonian_diag.get_hamiltonian(
                self.get_ft_potential_fast().tolist(),  # Takes abt 0.3s for a 10s run
                self._resolution,
                self.dz,
                self.mass,
                self.sho_omega,
                self.z_offset,
            )
        )

    def _calculate_off_diagonal_energies(self) -> NDArray:

        n_coordinates = len(self.coordinates)
        hamiltonian = np.zeros(shape=(n_coordinates, n_coordinates))

        for (index1, [nkx1, nky1, nz1]) in enumerate(self.coordinates):
            for (index2, [nkx2, nky2, nz2]) in enumerate(self.coordinates):
                # Number of jumps in units of dkx for this matrix element

                # As k-> k+ Nx * dkx the ft potential is left unchanged
                # Therefore we 'wrap round' ndkx into the region 0<ndkx<Nx
                # Where Nx is the number of x points

                # In reality we want to make sure ndkx < Nx (ie we should
                # make sure we generate enough points in the interpolation)
                ndkx = (nkx2 - nkx1) % self.Nx
                ndky = (nky2 - nky1) % self.Ny

                hamiltonian[index1, index2] = self._calculate_off_diagonal_entry(
                    nz1, nz2, ndkx, ndky
                )

        return hamiltonian

    def calculate_wavefunction(
        self, points: ArrayLike, eigenvector: Iterable[float]
    ) -> NDArray:
        points = np.array(points)
        out = np.zeros(shape=(points.shape[0]), dtype=complex)
        for (e, [nkx, nky, nz]) in zip(eigenvector, self.coordinates):
            out += (
                e
                * calculate_sho_wavefunction(
                    points[:, 2], self.sho_omega, self.mass, nz
                )
                * np.exp(1j * nkx * self.dkx * points[:, 0])
                * np.exp(1j * nky * self.dky * points[:, 1])
            )
        return out


def calculate_energy_eigenstates(
    hamiltonian: SurfaceHamiltonian, kx_points: NDArray, ky_points: NDArray
) -> EnergyEigenstates:
    eigenvalues = [
        hamiltonian.eigenvalues(round(abs(kx)), round(abs(ky)))
        for (kx, ky) in zip(kx_points, ky_points)
    ]
    arg_min_eigenvalue = [np.argmin(e) for e in eigenvalues]

    eigenvectors = [
        hamiltonian.eigenvectors(round(abs(kx)), round(abs(ky)))[
            arg_min_eigenvalue
        ].tolist()
        for (kx, ky) in zip(kx_points, ky_points)
    ]
    return {
        "kx_points": kx_points.tolist(),
        "ky_points": ky_points.tolist(),
        "resolution": hamiltonian._resolution,
        "eigenvalues": [e[arg] for (arg, e) in zip(arg_min_eigenvalue, eigenvalues)],
        "eigenvectors": eigenvectors,
    }


if __name__ == "__main__":
    data: EnergyInterpolation = {
        "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        "delta_x": 2 * np.pi * hbar,
        "delta_y": 2 * np.pi * hbar,
        "dz": 1,
    }
    config: SHOConfig = {"mass": 1, "sho_omega": 1 / hbar, "z_offset": 0}
    hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)

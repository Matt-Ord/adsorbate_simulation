import unittest
from typing import Literal

import hamiltonian_generator
import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis.basis import PositionBasis
from surface_potential_analysis.sho_basis import (
    SHOBasisConfig,
    calculate_sho_wavefunction,
    infinate_sho_basis_from_config,
)


class SHOBasisTest(unittest.TestCase):
    def test_sho_normalization(self) -> None:
        mass = hbar**2
        sho_omega = 1 / hbar
        x_points = np.linspace(-10, 10, 1001)

        for iz1 in range(12):
            for iz2 in range(12):
                sho_1 = calculate_sho_wavefunction(x_points, mass, sho_omega, iz1)
                sho_2 = calculate_sho_wavefunction(x_points, mass, sho_omega, iz2)
                sho_norm = (x_points[1] - x_points[0]) * np.sum(
                    sho_1 * sho_2, dtype=float
                )

                if iz1 == iz2:
                    self.assertAlmostEqual(sho_norm, 1.0)
                else:
                    self.assertAlmostEqual(sho_norm, 0.0)

    def test_calculate_sho_wavefunction(self) -> None:
        mass = hbar**2
        sho_omega = 1 / hbar
        z_points = np.linspace(-10, 10, np.random.randint(0, 1000))

        norm = np.sqrt(mass * sho_omega / hbar)

        phi_0_norm = np.sqrt(norm / np.sqrt(np.pi))
        phi_0_expected = phi_0_norm * np.exp(-((z_points * norm) ** 2) / 2)
        phi_0_actual = calculate_sho_wavefunction(z_points, sho_omega, mass, 0)

        np.testing.assert_allclose(phi_0_expected, phi_0_actual)

        phi_1_norm = np.sqrt(2 * norm / np.sqrt(np.pi))
        phi_1_expected = phi_1_norm * z_points * np.exp(-((z_points * norm) ** 2) / 2)
        phi_1_actual = calculate_sho_wavefunction(z_points, sho_omega, mass, 1)

        np.testing.assert_allclose(phi_1_expected, phi_1_actual)

        phi_2_norm = np.sqrt(norm / (2 * np.sqrt(np.pi)))
        phi_2_poly = (2 * z_points**2 - 1) * np.exp(-((z_points * norm) ** 2) / 2)
        phi_2_expected = phi_2_norm * phi_2_poly
        phi_2_actual = calculate_sho_wavefunction(z_points, sho_omega, mass, 2)

        np.testing.assert_allclose(phi_2_expected, phi_2_actual)

        phi_3_norm = np.sqrt(norm / (3 * np.sqrt(np.pi)))
        phi_3_poly = (2 * z_points**3 - 3 * z_points) * np.exp(
            -((z_points * norm) ** 2) / 2
        )
        phi_3_expected = phi_3_norm * phi_3_poly
        phi_3_actual = calculate_sho_wavefunction(z_points, sho_omega, mass, 3)

        np.testing.assert_allclose(phi_3_expected, phi_3_actual)

    def test_get_sho_rust(self) -> None:
        mass = hbar**2 * np.random.rand(1).item(0)
        sho_omega = np.random.rand(1).item(0) / hbar
        z_points = np.linspace(
            -20 * np.random.rand(1).item(0), 20 * np.random.rand(1).item(0), 1000
        )

        for n in range(14):
            actual = hamiltonian_generator.get_sho_wavefunction(
                z_points.tolist(), sho_omega, mass, n
            )
            expected = calculate_sho_wavefunction(z_points, sho_omega, mass, n)

            np.testing.assert_allclose(actual, expected)

    def test_infinate_sho_basis_from_config_normalization(self) -> None:
        nz = 12
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -10]),
        }
        parent: PositionBasis[Literal[1001]] = {
            "_type": "position",
            "delta_x": np.array([0, 0, 20]),
            "n": 1001,
        }
        basis = infinate_sho_basis_from_config(parent, config, 12)
        np.testing.assert_almost_equal(
            np.ones((nz,)), np.sum(basis["vectors"] * np.conj(basis["vectors"]), axis=1)
        )
"""Utility functions to help with the generation of a sho basis."""

import math
from typing import TYPE_CHECKING, Any, Literal, TypedDict, TypeVar

import numpy as np
import scipy
import scipy.special
from scipy.constants import hbar

from surface_potential_analysis.basis.basis import (
    BasisUtil,
    BasisWithLength,
    ExplicitBasis,
    FundamentalBasis,
    PositionBasis,
)
from surface_potential_analysis.basis_config.potential_basis import (
    PotentialBasisConfig,
    get_potential_basis_config_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis_config.basis_config import PositionBasisConfig

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)


def calculate_sho_wavefunction(
    x_points: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]],
    sho_omega: float,
    mass: float,
    n: int,
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]:
    """
    Calculate the value of a sho wavefunction at x_points.

    Parameters
    ----------
    x_points : np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
        Points to evaluate
    sho_omega : float
        Omega in 1/2 m omega ** 2 x **2
    mass : float
        Mass of the oscillator
    n : int
        index of the wavefunction to calculate

    Returns
    -------
    np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
        A list of values for the wavefunction in position basis
    """
    norm = (sho_omega * mass / hbar) ** 0.5
    normalized_x = x_points * norm

    prefactor = math.sqrt((norm / (2**n)) / (math.factorial(n) * math.sqrt(math.pi)))
    hermite = scipy.special.eval_hermite(n, normalized_x)
    exponential = np.exp(-np.square(normalized_x) / 2)
    return prefactor * hermite * exponential  # type: ignore[no-any-return]


def calculate_x_distances(
    parent: BasisWithLength[_L0Inv, Any],
    x_origin: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]:
    """Given a basis, calculate x distances with a projected value of zero at x_origin."""
    util = BasisUtil(parent)
    x_points = util.fundamental_x_points

    x0_norm = util.delta_x.copy() / np.linalg.norm(util.delta_x)
    distances_origin = np.dot(x0_norm, x_origin)
    return np.dot(x0_norm, x_points) + distances_origin  # type: ignore[no-any-return]


class SHOBasisConfig(TypedDict):
    """Configuration for a SHO Basis."""

    sho_omega: float
    mass: float
    x_origin: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]


def get_sho_potential_basis_config(
    parent: FundamentalBasis[_L0Inv], config: SHOBasisConfig, n: _L1Inv
) -> PotentialBasisConfig[_L0Inv, _L1Inv]:
    """
    Get a potential basis config assuming a SHO oscillator.

    Parameters
    ----------
    parent : FundamentalBasis[_L0Inv]
    config : SHOBasisConfig
    n : _L1Inv

    Returns
    -------
    PotentialBasisConfig[_L0Inv, _L1Inv]
    """
    util = BasisUtil(parent)

    delta_x1 = (
        np.array([0, 1, 0])
        if np.allclose([1, 0, 0], parent["delta_x"])
        else np.array([1, 0, 0])
    )
    delta_x2 = np.cross(parent["delta_x"], delta_x1)
    delta_x2 /= np.linalg.norm(delta_x2)

    potential_basis: PositionBasisConfig[_L0Inv, Literal[1], Literal[1]] = (
        util.get_fundamental_basis_in("position"),
        {"_type": "position", "delta_x": delta_x1, "n": 1},
        {"_type": "position", "delta_x": delta_x2, "n": 1},
    )
    x_distances = calculate_x_distances(parent, config["x_origin"])

    return {
        "potential": {
            "basis": potential_basis,
            "points": 0.5
            * config["mass"]
            * config["sho_omega"] ** 2
            * np.array([[x_distances]]) ** 2,
        },
        "n": n,
        "mass": config["mass"],
    }


def sho_basis_from_config(
    parent: FundamentalBasis[_L1Inv], config: SHOBasisConfig, n: _L0Inv
) -> ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]:
    """
    Calculate the exact sho basis for a given basis, by directly diagonalizing the sho wavefunction in this basis.

    The resulting wavefunction
    is guaranteed to be orthonormal in this basis.

    Parameters
    ----------
    parent : _FBInv
    config : SHOBasisConfig
    n : _L0Inv

    Returns
    -------
    ExplicitBasis[_L0Inv, _FBInv]
    """
    potential_basis_config = get_sho_potential_basis_config(parent, config, n)
    return get_potential_basis_config_basis(potential_basis_config)


def infinate_sho_basis_from_config(
    parent: PositionBasis[_L1Inv], config: SHOBasisConfig, n: _L0Inv
) -> ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]:
    """
    Generate an explicit sho basis assuming an infinate extent in the z direction.

    Parameters
    ----------
    parent : PositionBasis[_L1Inv]
    config : SHOBasisConfig
    n : _L0Inv

    Returns
    -------
    ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]
    """
    x_distances = calculate_x_distances(parent, config["x_origin"])

    vectors = np.array(
        [
            calculate_sho_wavefunction(
                x_distances, n=i, mass=config["mass"], sho_omega=config["sho_omega"]
            )
            for i in range(n)
        ]
    )
    util = BasisUtil(parent)
    normalized = vectors * np.sqrt(np.linalg.norm(util.fundamental_dx))
    return {"_type": "explicit", "parent": parent, "vectors": normalized}
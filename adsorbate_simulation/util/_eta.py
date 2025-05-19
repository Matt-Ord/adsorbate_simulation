from __future__ import annotations


def gamma_from_eta(eta: float, mass: float) -> float:
    """Get the damping coefficient from the friction and mass."""
    return eta / (mass * 2)


def eta_from_gamma(gamma: float, mass: float) -> float:
    """Get the friction coefficient from the damping and mass."""
    return gamma * mass * 2

from __future__ import annotations

import numpy as np
from surface_potential_analysis.basis_config.basis_config import BasisConfigUtil
from surface_potential_analysis.eigenstate.conversion import convert_eigenstate_to_basis
from surface_potential_analysis.wavepacket.wavepacket import get_eigenstate

from .s4_wavepacket import load_copper_wavepacket


def calculate_wavepacket_maximums() -> None:
    """Calculate the maximum of the k=0 eigenstate of a wavepacket for each band."""
    for band in range(20):
        wavepacket = load_copper_wavepacket(band)
        util = BasisConfigUtil(wavepacket["basis"])
        position_basis = util.get_fundamental_basis_in("position")
        util = BasisConfigUtil(position_basis)

        eigenstate = get_eigenstate(wavepacket, 0)
        converted = convert_eigenstate_to_basis(eigenstate, position_basis)

        print(f"Band {band}")  # noqa: T201
        print(util.get_stacked_index(np.argmax(converted["vector"])))  # noqa: T201

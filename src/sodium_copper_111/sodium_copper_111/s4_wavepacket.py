from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from surface_potential_analysis.util.decorators import npy_cached
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    generate_n_band_wavepacket,
    generate_wavepacket,
)

from sodium_copper_111.s2_hamiltonian import get_hamiltonian
from sodium_copper_111.surface_data import get_data_path

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import FundamentalMomentumAxis1d
    from surface_potential_analysis.hamiltonian.hamiltonian import Hamiltonian

    _SodiumWavepacket = Wavepacket[
        tuple[Literal[12]], tuple[FundamentalMomentumAxis1d[Literal[1000]]]
    ]


def _hamiltonian_generator(
    bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float_]]
) -> Hamiltonian[tuple[FundamentalMomentumAxis1d[Literal[1000]]]]:
    return get_hamiltonian(shape=(1000,), bloch_fraction=bloch_fraction)


@npy_cached(get_data_path("wavepacket.npy"), allow_pickle=True)
def get_all_wavepackets() -> list[_SodiumWavepacket]:
    save_bands = np.arange(99)
    return generate_wavepacket(
        _hamiltonian_generator,
        shape=(12,),
        save_bands=save_bands,
    )


def get_wavepacket(band: int = 0) -> _SodiumWavepacket:
    return get_all_wavepackets()[band]


@npy_cached(get_data_path("2_wavepacket.npy"), allow_pickle=True)
def get_n_band_wavepacket() -> (
    Wavepacket[tuple[Literal[24]], tuple[FundamentalMomentumAxis1d[Literal[1000]]]]
):
    return generate_n_band_wavepacket(_hamiltonian_generator, (12,), 4)

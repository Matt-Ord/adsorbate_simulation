from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis_like import BasisLike
from surface_potential_analysis.axis.stacked_axis import StackedBasis, StackedBasisLike
from surface_potential_analysis.operator.operator_list import OperatorList

if TYPE_CHECKING:
    from surface_potential_analysis.wavepacket.wavepacket import WavepacketList

_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_B2 = TypeVar("_B2", bound=BasisLike[Any, Any])

LocalizationOperator = OperatorList[_B0, _B1, _B2]
"""A list of operators, acting on each bloch k"""

_SB0 = TypeVar("_SB0", bound=StackedBasisLike[*tuple[Any, ...]])
_SB1 = TypeVar("_SB1", bound=StackedBasisLike[*tuple[Any, ...]])


def get_localized_wavepackets(
    wavepackets: WavepacketList[_B2, _SB1, _SB0],
    operator: LocalizationOperator[_SB1, _B1, _B2],
) -> WavepacketList[_B1, _SB1, _SB0]:
    """
    Apply the LocalizationOperator to produce localized wavepackets.

    Parameters
    ----------
    wavepackets : WavepacketList[_B2, _SB1, _SB0]
        The unlocalized wavepackets
    operator : LocalizationOperator[_SB1, _B1, _B2]
        The operator used to localize the wavepackets

    Returns
    -------
    WavepacketList[_B1, _SB1, _SB0]
        The localized wavepackets
    """
    stacked_operator = operator["data"].reshape(-1, *operator["basis"][1].shape)
    coefficients = np.moveaxis(stacked_operator, 0, -1)
    vectors = wavepackets["data"].reshape(*wavepackets["basis"][0].shape, -1)

    data = np.sum(
        coefficients[:, :, :, np.newaxis] * vectors[np.newaxis, :, :, :], axis=(1)
    )
    return {
        "basis": StackedBasis(
            StackedBasis(operator["basis"][1][0], wavepackets["basis"][0][1]),
            wavepackets["basis"][1],
        ),
        "data": data.reshape(-1),
    }
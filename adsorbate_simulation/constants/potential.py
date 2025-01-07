from __future__ import annotations

from scipy.constants import Avogadro, electron_volt  # type: ignore unknown

from adsorbate_simulation.system._potential import (
    CosPotential,
    FCCPotential,
    FreePotential,
)

LI_CU_TF_ENERGY = -(477.16) * 1e3 / Avogadro
LI_CU_BR_ENERGY = -(471.41) * 1e3 / Avogadro
LI_CU_TP_ENERGY = -(467.42) * 1e3 / Avogadro


LI_CU_DAVID_TOP_ENERGY = 45e-3 * electron_volt

LI_CU_COS_POTENTIAL = CosPotential(
    barrier_height=LI_CU_DAVID_TOP_ENERGY / 9,
)
"""An effective 1d potential for a Lithium-Copper system."""

LI_CU_111_2D_POTENTIAL = FCCPotential(
    top_site_energy=LI_CU_DAVID_TOP_ENERGY,
)

LI_CU_COS_POTENTIAL_ELENA = CosPotential(
    barrier_height=LI_CU_DAVID_TOP_ENERGY,
)
"""The potential used in the Elena's paper for a Lithium-Copper system."""


NA_CU_TF_ENERGY = -(416.78) * 1e3 / Avogadro
NA_CU_BR_ENERGY = -(414.24) * 1e3 / Avogadro
NA_CU_TP_ENERGY = -(431.79) * 1e3 / Avogadro

NA_CU_DAVID_TOP_ENERGY = 55e-3 * electron_volt

NA_CU_COS_POTENTIAL = CosPotential(
    barrier_height=NA_CU_DAVID_TOP_ENERGY / 9,
)
"""An effective 1d potential for a Sodium-Copper system."""

NA_CU_111_2D_POTENTIAL = FCCPotential(
    top_site_energy=NA_CU_DAVID_TOP_ENERGY,
)

NA_CU_COS_POTENTIAL_ELENA = CosPotential(
    barrier_height=NA_CU_DAVID_TOP_ENERGY,
)
"""The potential used in the Elena's paper for a Sodium-Copper system."""

K_CU_TF_ENERGY = -(334.09) * 1e3 / Avogadro
K_CU_BR_ENERGY = -(332.58) * 1e3 / Avogadro
K_CU_TP_ENERGY = -(378.59) * 1e3 / Avogadro


FREE_POTENTIAL = FreePotential()

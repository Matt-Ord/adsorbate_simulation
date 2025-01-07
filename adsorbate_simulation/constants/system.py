from __future__ import annotations

from adsorbate_simulation.constants.lattice import (
    CU_111_1D_UNIT_CELL,
    CU_111_2D_UNIT_CELL,
    DIMENSIONLESS_UNIT_CELL,
)
from adsorbate_simulation.constants.mass import (
    DIMENSIONLESS_MASS,
    LITHIUM_MASS,
    SODIUM_MASS,
)
from adsorbate_simulation.constants.potential import (
    FREE_POTENTIAL,
    LI_CU_111_2D_POTENTIAL,
    LI_CU_COS_POTENTIAL,
    NA_CU_111_2D_POTENTIAL,
    NA_CU_COS_POTENTIAL,
)
from adsorbate_simulation.system._system import System

DIMENSIONLESS_1D_SYSTEM = System(
    DIMENSIONLESS_MASS,
    FREE_POTENTIAL,
    DIMENSIONLESS_UNIT_CELL,
)

LI_CU_111_1D_SYSTEM = System(
    LITHIUM_MASS,
    LI_CU_COS_POTENTIAL,
    CU_111_1D_UNIT_CELL,
)

LI_CU_111_2D_SYSTEM = System(
    LITHIUM_MASS,
    LI_CU_111_2D_POTENTIAL,
    CU_111_2D_UNIT_CELL,
)

NA_CU_111_1D_SYSTEM = System(
    SODIUM_MASS,
    NA_CU_COS_POTENTIAL,
    CU_111_1D_UNIT_CELL,
)

NA_CU_111_2D_SYSTEM = System(
    SODIUM_MASS,
    NA_CU_111_2D_POTENTIAL,
    CU_111_2D_UNIT_CELL,
)

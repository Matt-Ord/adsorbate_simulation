from __future__ import annotations

import numpy as np
from slate.metadata import AxisDirections

from adsorbate_simulation.system._basis import SimulationCell

CELL_DIRECTIONS_1D = AxisDirections(vectors=(np.array([1]),))

CELL_DIRECTIONS_2D_111 = AxisDirections(
    vectors=(np.array([1, 0]), (np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)])))
)
CELL_DIRECTIONS_2D_100 = AxisDirections(vectors=(np.array([1, 0]), (np.array([0, 1]))))


CU_111_LATTICE_CONSTANT = 3.615e-10 / np.sqrt(2)

CU_111_1D_UNIT_CELL = SimulationCell(
    lengths=((1 / np.sqrt(6)) * CU_111_LATTICE_CONSTANT,),
    directions=CELL_DIRECTIONS_1D,
)
"""The unit cell for a 1D system of copper."""

CU_111_2D_UNIT_CELL = SimulationCell(
    lengths=(CU_111_LATTICE_CONSTANT, CU_111_LATTICE_CONSTANT),
    directions=CELL_DIRECTIONS_2D_111,
)
"""The unit cell for a 2D system of copper."""

CU_1D_UNIT_CELL_ELENA = SimulationCell(
    lengths=(np.sqrt(3 / 2) * CU_111_LATTICE_CONSTANT,),
    directions=CELL_DIRECTIONS_1D,
)
"""The unit cell used in Elena's paper for a 1D system of copper."""


DIMENSIONLESS_UNIT_CELL = SimulationCell(
    lengths=(2 * np.pi,),
    directions=CELL_DIRECTIONS_1D,
)

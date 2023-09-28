from __future__ import annotations

from hydrogen_nickel_111.s4_wavepacket import get_all_wavepackets_hydrogen
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    get_all_wavepackets_hydrogen()


if __name__ == "__main__":
    main()

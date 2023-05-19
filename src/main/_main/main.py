from __future__ import annotations

import nickel_111
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    nickel_111.s5_overlap_analysis.print_averaged_overlap_nickel()


if __name__ == "__main__":
    main()

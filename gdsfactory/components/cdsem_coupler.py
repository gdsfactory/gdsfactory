"""CD SEM structures."""
from functools import partial
from typing import Tuple

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.coupler_straight import coupler_straight
from gdsfactory.cross_section import strip
from gdsfactory.grid import grid
from gdsfactory.types import CrossSectionFactory


@cell
def cdsem_coupler(
    width: float = 0.45,
    length: float = 420.0,
    gaps: Tuple[float, ...] = (0.15, 0.2, 0.25),
    cross_section: CrossSectionFactory = strip,
) -> Component:
    """Returns 2 coupled waveguides gap sweep.

    Args:
        width: for the waveguide
        length: for the line
        gaps: list of gaps
        cross_section

    """

    cross_section = partial(cross_section, width=width)
    couplers = [
        coupler_straight(length=length, gap=gap, cross_section=cross_section)
        for gap in gaps
    ]
    return grid(couplers)


if __name__ == "__main__":
    c = cdsem_coupler()
    c.show()

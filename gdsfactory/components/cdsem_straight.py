"""CD SEM structures."""
from functools import partial
from typing import Tuple

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.cross_section import strip
from gdsfactory.grid import grid
from gdsfactory.types import CrossSectionFactory

LINE_LENGTH = 420.0


@cell
def cdsem_straight(
    widths: Tuple[float, ...] = (0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
    length: float = LINE_LENGTH,
    cross_section: CrossSectionFactory = strip,
) -> Component:
    """Returns straight waveguide lines width sweep.

    Args:
        widths: for the sweep
        length: for the line
        cross_section:
    """

    lines = []
    for width in widths:
        cross_section = partial(cross_section, width=width)
        lines.append(straight_function(length=length, cross_section=cross_section))

    return grid(lines)


if __name__ == "__main__":
    c = cdsem_straight()
    c.show()

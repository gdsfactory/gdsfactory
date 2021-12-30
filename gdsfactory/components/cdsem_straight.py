"""CD SEM structures."""
from functools import partial
from typing import Optional, Tuple

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.cross_section import strip
from gdsfactory.grid import grid
from gdsfactory.types import ComponentFactory, CrossSectionFactory

text_rectangular_mini = partial(text_rectangular, size=1)

LINE_LENGTH = 420.0


@cell
def cdsem_straight(
    widths: Tuple[float, ...] = (0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
    length: float = LINE_LENGTH,
    cross_section: CrossSectionFactory = strip,
    text: Optional[ComponentFactory] = text_rectangular_mini,
    spacing: float = 3,
) -> Component:
    """Returns straight waveguide lines width sweep.

    Args:
        widths: for the sweep
        length: for the line
        cross_section: for the lines
        text: optional text for labels
        spacing: edge to edge spacing
    """

    lines = []
    for width in widths:
        cross_section = partial(cross_section, width=width)
        line = straight_function(length=length, cross_section=cross_section)
        if text:
            line = line.copy()
            t = line << text(str(int(width * 1e3)))
            t.xmin = line.xmax + 5
            t.y = 0
        lines.append(line)

    return grid(lines, spacing=(0, spacing))


if __name__ == "__main__":
    c = cdsem_straight()
    c.show()

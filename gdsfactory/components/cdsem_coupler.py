"""CD SEM structures."""
from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.coupler_straight import coupler_straight
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.grid import grid
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

text_rectangular_mini = partial(text_rectangular, size=1)


@cell
def cdsem_coupler(
    width: float = 0.45,
    length: float = 420.0,
    gaps: Tuple[float, ...] = (0.15, 0.2, 0.25),
    cross_section: CrossSectionSpec = "strip",
    text: Optional[ComponentSpec] = text_rectangular_mini,
    spacing: float = 3,
) -> Component:
    """Returns 2 coupled waveguides gap sweep.

    Args:
        width: for the waveguide.
        length: for the line.
        gaps: list of gaps for the sweep.
        cross_section: for the lines.
        text: optional text for labels.
        spacing: edge to edge spacing.
    """
    cross_section = gf.get_cross_section(cross_section, width=width)

    couplers = []

    for gap in gaps:
        coupler = coupler_straight(length=length, gap=gap, cross_section=cross_section)
        if text:
            coupler = coupler.copy()
            t = coupler << gf.get_component(text, text=str(int(gap * 1e3)))
            t.xmin = coupler.xmax + 5
            t.y = 0

        couplers.append(coupler)

    return grid(couplers, spacing=(0, spacing))


if __name__ == "__main__":
    c = cdsem_coupler()
    c.show(show_ports=True)

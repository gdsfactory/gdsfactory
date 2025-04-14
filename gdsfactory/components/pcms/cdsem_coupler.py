"""CD SEM structures."""

from __future__ import annotations

from collections.abc import Sequence

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name
def cdsem_coupler(
    length: float = 420.0,
    gaps: Sequence[float] = (0.15, 0.2, 0.25),
    cross_section: CrossSectionSpec = "strip_no_ports",
    text: ComponentSpec | None = "text_rectangular",
    spacing: float = 7.0,
    positions: Sequence[float | None] | None = None,
    width: float | None = None,
    text_size: float = 1.0,
) -> Component:
    """Returns 2 coupled waveguides gap sweep.

    Args:
        length: for the line.
        gaps: list of gaps for the sweep.
        cross_section: for the lines.
        text: optional text for labels.
        spacing: Optional center to center spacing.
        positions: Optional positions for the text labels.
        width: width of the waveguide. If None, it will use the width of the cross_section.
        text_size: size of the text.
    """
    c = Component()
    if width:
        xs = gf.get_cross_section(cross_section, width=width)
    else:
        xs = gf.get_cross_section(cross_section)
    p = 0.0

    if positions is not None:
        positions = positions or [None] * len(gaps)
    else:
        positions = [i * spacing for i in range(len(gaps))]

    for gap, position in zip(gaps, positions):
        line = c << gf.c.coupler_straight(length=length, cross_section=xs, gap=gap)
        p = position or p
        line.ymin = p
        if text:
            t = c << gf.get_component(text, text=str(int(gap * 1e3)), size=text_size)
            t.xmin = line.xmax + 5
            t.ymin = p

    return c


if __name__ == "__main__":
    # c = cdsem_coupler(cross_section="rib_with_trenches")
    # c = cdsem_coupler(cross_section="strip", width=1)
    c = cdsem_coupler()
    c.show()

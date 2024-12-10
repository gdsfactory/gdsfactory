"""CD SEM structures."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components import coupler_straight, text_rectangular
from gdsfactory.typings import ComponentFactory, CrossSectionSpec

text_rectangular_mini = partial(text_rectangular, size=1)


@gf.cell
def cdsem_coupler(
    length: float = 420.0,
    gaps: Sequence[float] = (0.15, 0.2, 0.25),
    cross_section: CrossSectionSpec = "strip",
    text: ComponentFactory | None = text_rectangular_mini,
    spacing: float = 7.0,
    positions: Sequence[float | None] | None = None,
    width: float | None = None,
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
    """
    c = Component()
    xs = gf.get_cross_section(cross_section, width=width)
    p = 0

    if positions is not None:
        positions = positions or [None] * len(gaps)
    else:
        positions = [i * spacing for i in range(len(gaps))]

    for gap, position in zip(gaps, positions):
        line = c << coupler_straight(length=length, cross_section=xs, gap=gap)
        p = position or p  # type: ignore
        line.dymin = p
        if text:
            t = c << text(str(int(gap * 1e3)))
            t.dxmin = line.dxmax + 5
            t.dymin = p

    return c


if __name__ == "__main__":
    # c = cdsem_coupler(cross_section="rib_with_trenches")
    c = cdsem_coupler(cross_section="strip", width=1)
    c.show()

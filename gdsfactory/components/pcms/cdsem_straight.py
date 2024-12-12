"""CD SEM structures."""

from __future__ import annotations

from collections.abc import Sequence

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

LINE_LENGTH = 420.0


@gf.cell
def cdsem_straight(
    widths: Sequence[float] = (0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
    length: float = LINE_LENGTH,
    cross_section: CrossSectionSpec = "strip",
    text: ComponentSpec | None = "text_rectangular",
    spacing: float = 7.0,
    positions: Sequence[float | None] | None = None,
    text_size: float = 1,
) -> Component:
    """Returns straight waveguide lines width sweep.

    Args:
        widths: for the sweep.
        length: for the line.
        cross_section: for the lines.
        text: optional text for labels.
        spacing: Optional center to center spacing.
        positions: Optional positions for the text labels.
        text_size: in um.
    """
    c = Component()
    p = 0
    if positions is not None:
        positions = positions or [None] * len(widths)
    else:
        positions = [i * spacing for i in range(len(widths))]

    for width, position in zip(widths, positions):
        line = c << gf.c.straight(
            length=length, cross_section=cross_section, width=width
        )
        p = position or p  # type: ignore
        line.dymin = p
        if text:
            t = c << gf.get_component(text, text=str(int(width * 1e3)), size=text_size)
            t.dxmin = line.dxmax + 5
            t.dymin = p

    return c


if __name__ == "__main__":
    c = cdsem_straight()
    c.show()

"""CD SEM structures."""

from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory import cell
from gdsfactory.component import Component
from gdsfactory.components.straight import straight
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

LINE_LENGTH = 420.0


@cell
def cdsem_straight(
    widths: tuple[float, ...] = (0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
    length: float = LINE_LENGTH,
    cross_section: CrossSectionSpec = "strip",
    text: ComponentSpec | None = "text_rectangular_mini",
    spacing: float | None = 7.0,
    positions: tuple[float, ...] | None = None,
) -> Component:
    """Returns straight waveguide lines width sweep.

    Args:
        widths: for the sweep.
        length: for the line.
        cross_section: for the lines.
        text: optional text for labels.
        spacing: Optional center to center spacing.
        positions: Optional positions for the text labels.
    """
    c = Component()
    p = 0

    if positions is None and spacing is None:
        raise ValueError("Either positions or spacing should be defined")
    elif positions:
        positions = positions or [None] * len(widths)
    else:
        positions = np.arange(len(widths)) * spacing

    for width, position in zip(widths, positions):
        line = c << straight(length=length, cross_section=cross_section, width=width)
        p = position or p
        line.dymin = p
        if text:
            t = c << gf.get_component(text, text=str(int(width * 1e3)))
            t.dxmin = line.dxmax + 5
            t.dymin = p

    return c


if __name__ == "__main__":
    c = cdsem_straight()
    c.show()

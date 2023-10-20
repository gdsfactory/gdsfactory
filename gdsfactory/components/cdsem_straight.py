"""CD SEM structures."""
from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.straight import straight
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.typings import ComponentFactory, CrossSectionSpec

text_rectangular_mini = partial(text_rectangular, size=1)

LINE_LENGTH = 420.0


@cell
def cdsem_straight(
    widths: tuple[float, ...] = (0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
    length: float = LINE_LENGTH,
    cross_section: CrossSectionSpec = "xs_sc",
    text: ComponentFactory | None = text_rectangular_mini,
    spacing: float | None = 7.0,
    positions: tuple[float, ...] | None = None,
    **kwargs,
) -> Component:
    """Returns straight waveguide lines width sweep.

    Args:
        widths: for the sweep.
        length: for the line.
        cross_section: for the lines.
        text: optional text for labels.
        spacing: Optional center to center spacing.
        positions: Optional positions for the text labels.
        kwargs: cross_section settings.
    """
    c = Component()
    p = 0

    xs = gf.get_cross_section(cross_section, **kwargs)

    if positions is None and spacing is None:
        raise ValueError("Either positions or spacing should be defined")
    elif positions:
        positions = positions or [None] * len(widths)
    else:
        positions = np.arange(len(widths)) * spacing

    for width, position in zip(widths, positions):
        line = c << straight(length=length, cross_section=xs, width=width)
        p = position or p
        line.ymin = p
        if text:
            t = c << text(text=str(int(width * 1e3)))
            t.xmin = line.xmax + 5
            t.y = p

    return c


if __name__ == "__main__":
    c = cdsem_straight()
    c.show(show_ports=True)

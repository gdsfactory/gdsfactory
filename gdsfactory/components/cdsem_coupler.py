"""CD SEM structures."""
from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.coupler_straight import coupler_straight
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.typings import ComponentFactory, CrossSectionSpec

text_rectangular_mini = partial(text_rectangular, size=1)


@cell
def cdsem_coupler(
    length: float = 420.0,
    gaps: tuple[float, ...] = (0.15, 0.2, 0.25),
    cross_section: CrossSectionSpec = "xs_sc",
    text: ComponentFactory | None = text_rectangular_mini,
    spacing: float | None = 7.0,
    positions: tuple[float, ...] | None = None,
    **kwargs,
) -> Component:
    """Returns 2 coupled waveguides gap sweep.

    Args:
        length: for the line.
        gaps: list of gaps for the sweep.
        cross_section: for the lines.
        text: optional text for labels.
        spacing: Optional center to center spacing.
        positions: Optional positions for the text labels.
        kwargs: cross_section settings.
    """
    c = Component()
    xs = gf.get_cross_section(cross_section, **kwargs)
    p = 0

    if positions is None and spacing is None:
        raise ValueError("Either positions or spacing should be defined")
    elif positions:
        positions = positions or [None] * len(gaps)
    else:
        positions = np.arange(len(gaps)) * spacing

    for gap, position in zip(gaps, positions):
        line = c << coupler_straight(length=length, cross_section=xs, gap=gap)
        p = position or p
        line.ymin = p
        if text:
            t = c << text(str(int(gap * 1e3)))
            t.xmin = line.xmax + 5
            t.y = p

    return c


if __name__ == "__main__":
    c = cdsem_coupler(cross_section="xs_rc_with_trenches")
    c.show(show_ports=True)

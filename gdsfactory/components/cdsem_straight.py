"""CD SEM structures."""
from __future__ import annotations

from functools import partial

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.straight import straight
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

text_rectangular_mini = partial(text_rectangular, size=1)

LINE_LENGTH = 420.0


@cell
def cdsem_straight(
    widths: tuple[float, ...] = (0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
    length: float = LINE_LENGTH,
    cross_section: CrossSectionSpec = "strip",
    text: ComponentSpec | None = text_rectangular_mini,
    spacing: float = 5,
) -> Component:
    """Returns straight waveguide lines width sweep.

    Args:
        widths: for the sweep.
        length: for the line.
        cross_section: for the lines.
        text: optional text for labels.
        spacing: edge to edge spacing.
    """
    c = Component()

    ymin = 0

    for width in widths:
        line = c << straight(length=length, cross_section=cross_section, width=width)
        line.ymin = ymin
        if text:
            t = c << text(str(int(width * 1e3)))
            t.xmin = line.xmax + 5
            t.y = ymin

        ymin += width + spacing

    return c


if __name__ == "__main__":
    c = cdsem_straight()
    c.show(show_ports=True)

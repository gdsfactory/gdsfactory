"""CD SEM structures."""
from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

LINE_LENGTH = 420.0

text_rectangular_mini = partial(text_rectangular, size=1)


@cell
def cdsem_bend180(
    width: float = 0.5,
    radius: float = 10.0,
    wg_length: float = LINE_LENGTH,
    straight: ComponentSpec = "straight",
    bend90: ComponentSpec = "bend_circular",
    cross_section: CrossSectionSpec = "strip",
    text: ComponentSpec = text_rectangular_mini,
) -> Component:
    """Returns CDSEM structures.

    Args:
        width: of the line.
        radius: um.
        wg_length: in um.
        straight: spec.
        bend90: spec.
        cross_section: spec.
        text: spec.
    """
    c = Component()
    r = radius

    if wg_length is None:
        wg_length = 2 * r

    bend90 = gf.get_component(
        bend90, cross_section=cross_section, radius=r, width=width
    )
    wg = gf.get_component(
        straight, cross_section=cross_section, length=wg_length, width=width
    )

    # Add the U-turn on straight layer
    b1 = c.add_ref(bend90)
    b2 = c.add_ref(bend90)
    b2.connect("o2", b1.ports["o1"])

    wg1 = c.add_ref(wg)
    wg1.connect("o1", b1.ports["o2"])

    wg2 = c.add_ref(wg)
    wg2.connect("o1", b2.ports["o1"])

    label = c << text(text=str(int(width * 1e3)))
    label.ymax = b2.ymin - 5
    label.x = 0
    b1.rotate(90)
    b2.rotate(90)
    wg1.rotate(90)
    wg2.rotate(90)
    label.rotate(90)
    return c


if __name__ == "__main__":
    c = cdsem_bend180(width=2)
    c.show(show_ports=True)

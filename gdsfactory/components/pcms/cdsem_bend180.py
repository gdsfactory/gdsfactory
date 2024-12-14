"""CD SEM structures."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

LINE_LENGTH = 420.0


@gf.cell
def cdsem_bend180(
    width: float = 0.5,
    radius: float = 10.0,
    wg_length: float | None = LINE_LENGTH,
    straight: ComponentSpec = "straight",
    bend90: ComponentSpec = "bend_circular",
    cross_section: CrossSectionSpec = "strip",
    text: ComponentSpec = "text_rectangular",
    text_size: float = 1.0,
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
        text_size: um.
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

    label = c << gf.get_component(text, text=str(int(width * 1e3)), size=text_size)
    label.dymax = b2.dymin - 5
    label.dx = 0

    c2 = gf.Component()
    ref = c2 << c
    ref.drotate(90)
    c2.flatten()
    return c2


if __name__ == "__main__":
    c = cdsem_bend180(width=2)
    c.show()

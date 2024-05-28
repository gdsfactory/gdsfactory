from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_s import bend_s
from gdsfactory.components.straight import straight
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def coupler_asymmetric(
    bend: ComponentSpec = bend_s,
    gap: float = 0.234,
    dy: float = 2.5,
    dx: float = 10.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Bend coupled to straight waveguide.

    Args:
        bend: spec.
        gap: um.
        dy: port to port vertical spacing.
        dx: bend length in x direction.
        cross_section: spec.

    .. code::

                        dx
                     |-----|
                      _____ o2
                     /         |
               _____/          |
         gap o1____________    |  dy
                            o3
    """
    c = Component()
    x = gf.get_cross_section(cross_section)
    width = x.width
    bend_component = (
        bend(size=(dx, dy - gap - width), cross_section=cross_section)
        if callable(bend)
        else bend
    )
    wg = straight(cross_section=cross_section)

    w = bend_component.ports[0].dwidth
    y = (w + gap) / 2

    wg = c << wg
    bend = c << bend_component
    bend.dmirror_y()
    bend.dxmin = 0
    wg.dxmin = 0

    bend.dmovey(-y)
    wg.dmovey(+y)

    port_width = 2 * w + gap
    c.add_port(
        name="o1",
        center=(0, 0),
        width=port_width,
        orientation=180,
        cross_section=x,
    )
    c.add_port(name="o3", port=bend.ports[1])
    c.add_port(name="o2", port=wg.ports[0])

    c.flatten()
    return c


if __name__ == "__main__":
    c = coupler_asymmetric(gap=0.2)
    c.show()

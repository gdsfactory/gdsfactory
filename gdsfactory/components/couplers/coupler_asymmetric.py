from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, Delta


@gf.cell
def coupler_asymmetric(
    gap: float = 0.234,
    dy: Delta = 2.5,
    dx: Delta = 10.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Bend coupled to straight waveguide.

    Args:
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
    bend = gf.c.bend_s(size=(dx, dy - gap - width), cross_section=cross_section)
    wg = gf.c.straight(cross_section=cross_section)

    w = bend.ports[0].width
    y = (w + gap) / 2

    wg_ref = c << wg
    bend_ref = c << bend
    bend_ref.dmirror_y()
    bend_ref.dxmin = 0
    wg_ref.dxmin = 0

    bend_ref.dmovey(-y)
    wg_ref.dmovey(+y)

    port_width = 2 * w + gap
    c.add_port(
        name="o1",
        center=(0, 0),
        width=port_width,
        orientation=180,
        cross_section=x,
    )
    c.add_port(name="o3", port=bend_ref.ports[1])
    c.add_port(name="o2", port=wg_ref.ports[0])
    c.flatten()
    return c


if __name__ == "__main__":
    c = coupler_asymmetric(gap=0.2)
    c.show()

from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.path import arc
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import CrossSectionSpec


@gf.cell
def bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    npoints: int | None = None,
    cross_section: CrossSectionSpec = "xs_sc",
    add_pins: bool = True,
) -> Component:
    """Returns a radial arc.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: angle of arc (degrees).
        npoints: number of points.
        cross_section: spec (CrossSection, string or dict).
        add_pins: add pins to the component.

    .. code::

                  o2
                  |
                 /
                /
               /
       o1_____/
    """
    x = gf.get_cross_section(cross_section)
    radius = radius or x.radius

    p = arc(radius=radius, angle=angle, npoints=npoints)
    c = Component()
    path = p.extrude(x)
    ref = c << path
    c.add_ports(ref.ports)

    c.absorb(ref)
    c.info["length"] = float(snap_to_grid(p.length()))
    c.info["dy"] = snap_to_grid(float(abs(p.points[0][0] - p.points[-1][0])))
    c.info["radius"] = float(radius)
    x.add_bbox(c)
    if add_pins:
        x.add_pins(c)
    return c


bend_circular180 = partial(bend_circular, angle=180)


if __name__ == "__main__":
    from gdsfactory.generic_tech import get_generic_pdk

    PDK = get_generic_pdk()
    PDK.activate()

    c = bend_circular(
        angle=180,
        cross_section="xs_rc",
    )
    c.show(show_ports=True)

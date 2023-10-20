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
    layer: gf.typings.LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "xs_sc",
    add_pins: bool = True,
    add_bbox: callable | None = None,
) -> Component:
    """Returns a radial arc.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: angle of arc (degrees).
        npoints: number of points.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: spec (CrossSection, string or dict).
        add_pins: add pins to the component.
        add_bbox: optional function to add bounding box to the component.

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
    if layer or width:
        x = x.copy(layer=layer or x.layer, width=width or x.width)

    p = arc(radius=radius, angle=angle, npoints=npoints)
    c = Component()
    path = p.extrude(x)
    ref = c << path
    c.add_ports(ref.ports)
    c.absorb(ref)

    c.info["length"] = float(snap_to_grid(p.length()))
    c.info["dy"] = snap_to_grid(float(abs(p.points[0][0] - p.points[-1][0])))
    c.info["radius"] = float(radius)
    x.validate_radius(radius)

    top = None if int(angle) in {180, -180, -90} else 0
    bottom = 0 if int(angle) in {-90} else None
    if add_bbox:
        add_bbox(c)
    else:
        x.add_bbox(c, top=top, bottom=bottom)
    if add_pins:
        x.add_pins(c)
    c.add_route_info(
        cross_section=x, length=c.info["length"], n_bend_90=abs(angle / 90.0)
    )
    return c


bend_circular180 = partial(bend_circular, angle=180)


if __name__ == "__main__":
    from gdsfactory.generic_tech import get_generic_pdk

    PDK = get_generic_pdk()
    PDK.activate()

    c = bend_circular(
        angle=180,
        cross_section="xs_rc",
        layer=(2, 0),
    )
    c.show(show_ports=True)

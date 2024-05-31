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
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
) -> Component:
    """Returns a radial arc.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: angle of arc (degrees).
        npoints: number of points.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.

    .. code::

                  o2
                  |
                 /
                /
        o1_____/
    """
    x = gf.get_cross_section(cross_section)
    radius = radius or x.radius
    if layer or width:
        x = x.copy(layer=layer or x.layer, width=width or x.width)

    p = arc(radius=radius, angle=angle, npoints=npoints)
    c = p.extrude(x)

    c.info["length"] = float(snap_to_grid(p.length()))
    c.info["dy"] = float(abs(p.points[0][0] - p.points[-1][0]))
    c.info["radius"] = float(radius)
    if not allow_min_radius_violation:
        x.validate_radius(radius)
    c.add_route_info(
        cross_section=x,
        length=c.info["length"],
        n_bend_90=abs(angle / 90.0),
        min_bend_radius=radius,
    )
    return c


bend_circular180 = partial(bend_circular, angle=180)


if __name__ == "__main__":
    c = gf.Component()
    r = c << bend_circular(radius=5)
    # r.dmirror()
    c.show()

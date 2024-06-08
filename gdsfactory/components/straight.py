"""Straight waveguide."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.cross_section import CrossSectionSpec


@gf.cell
def straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string or dict).
        kwargs: additional cross_section arguments.

    .. code::

        o1 -------------- o2
                length
    """
    x = gf.get_cross_section(cross_section, **kwargs)
    p = gf.path.straight(length=length, npoints=npoints)
    c = p.extrude(x)
    x.add_bbox(c)

    c.info["length"] = length
    c.info["width"] = x.width if len(x.sections) == 0 else x.sections[0].width
    c.add_route_info(cross_section=x, length=length)
    return c


@gf.vcell
def straight_all_angle(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> ComponentAllAngle:
    """Returns a Straight waveguide with offgrid ports.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string or dict).
        kwargs: additional cross_section arguments.

    .. code::

        o1 -------------- o2
                length
    """
    x = gf.get_cross_section(cross_section, **kwargs)
    p = gf.path.straight(length=length, npoints=npoints)
    c = p.extrude(x, all_angle=True)
    x.add_bbox(c)

    c.info["length"] = length
    c.info["width"] = x.width if len(x.sections) == 0 else x.sections[0].width
    c.add_route_info(cross_section=x, length=length)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component()
    w = straight(
        length=10,
        cross_section="rib_bbox",
    )
    ref = c << w
    ref.dxmin = 10
    p = c.get_polygons_points()
    p = list(p.values())
    print(p[0][0])
    c.show()

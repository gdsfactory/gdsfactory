"""Straight waveguide."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
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
    c = Component()

    x = gf.get_cross_section(cross_section, **kwargs)
    p = gf.path.straight(length=length, npoints=npoints)
    path = p.extrude(x)
    ref = c << path
    c.add_ports(ref.ports)
    # x.apply_enclosure(c)
    x.add_bbox(c)

    c.info["length"] = length
    c.info["width"] = x.width if len(x.sections) == 0 else x.sections[0].width
    c.add_route_info(cross_section=x, length=length)
    c.flatten()
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component()
    w = straight(
        length=10,
        cross_section="strip",
    )
    ref = c << w
    ref.dxmin = 10
    c.show()

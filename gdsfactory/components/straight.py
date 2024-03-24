"""Straight waveguide."""

from __future__ import annotations

import warnings

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import CrossSectionSpec


@gf.cell
def straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "xs_sc",
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

    c.info["length"] = length
    c.info["width"] = x.width
    c.add_route_info(cross_section=x, length=length)
    c.absorb(ref)
    return c


@gf.cell
def straight_array(
    n: int = 4,
    spacing: float = 4.0,
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> Component:
    """Array of straights connected with grating couplers.

    useful to align the 4 corners of the chip

    Args:
        n: number of straights.
        spacing: edge to edge straight spacing.
        length: straight length (um).
        cross_section: specification (CrossSection, string or dict).
    """
    warnings.warn("Use gf.components.array(straight) instead", DeprecationWarning)

    c = Component()
    wg = straight(cross_section=cross_section, length=length)

    for i in range(n):
        wref = c.add_ref(wg)
        wref.y += i * (spacing + wg.info["width"])
        c.add_ports(wref.ports, prefix=str(i))

    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    # xs = gf.cross_section.strip(bbox_layers=[(111, 0)], bbox_offsets=[3])
    # c = straight(cross_section=xs, info=dict(simulation="eme"))
    c = straight(cross_section="xs_rc_bbox")
    # print(c.info["simulation"])
    # c = gf.Component()
    # ref = c << straight(width=3e-3, length=3e-3)
    # ref.xmin = 0
    # ref.ymin = 0
    # ref.center = (0, 0)

    c.show()

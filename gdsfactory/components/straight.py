"""Straight waveguide."""
from __future__ import annotations

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

    c.info["length"] = float(length)
    if len(x.sections) == 0:
        c.info["width"] = x.width
    else:
        c.info["width"] = x.sections[0].width

    c.add_route_info(cross_section=x, length=length)
    c.absorb(ref)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    # c = gf.Component()
    # ref = c << straight(cross_section="xs_rc")
    # ref2 = c << straight(cross_section="xs_rc")
    # ref2.center = ref.center + kdb.Point(0, 1000)
    # ref2.d.move((0, 10))
    # ref.name = "straight"
    # print(c.insts['straight'].ports)

    # xs = gf.cross_section.pn()
    # xs = xs.mirror()
    # c = straight(cross_section=xs)
    # gdspath = c.write_gds()
    c = straight(
        length=10,
        cross_section="xs_sc",
    )
    c.show()

"""Straight waveguide."""
from __future__ import annotations

from collections.abc import Callable

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import CrossSectionSpec
from gdsfactory.typings import Metadata


@gf.cell
def straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "xs_sc",
    post_process: Callable | None = None,
    info: Metadata | None = None,
    **kwargs,
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string or dict).
        post_process: function to post process the component.
        info: additional information to add to the component.
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
    x.add_bbox(c, right=0, left=0)
    if post_process:
        post_process(c)
    if info:
        c.info.update(info)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    xs = gf.cross_section.strip(bbox_layers=[(111, 0)], bbox_offsets=[3])
    c = straight(cross_section=xs, info=dict(simulation="eme"))
    # print(c.info["simulation"])
    # c = gf.Component()
    # ref = c << straight(width=3e-3, length=3e-3)
    # ref.xmin = 0
    # ref.ymin = 0
    # ref.center = (0, 0)

    c.show()
    # print(c.bbox)
    # c._repr_html_()
    # c.show()
    # c.show(show_ports=True)
    # print(c.bbox)

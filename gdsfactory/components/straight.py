"""Straight waveguide."""
from __future__ import annotations

import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.component import Component
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import CrossSectionSpec


@gf.cell
def straight(
    length: float = 10.0,
    npoints: int = 2,
    with_bbox: bool = True,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        kwargs: cross_section settings.

    .. code::

        o1 -------------- o2
                length
    """
    length = snap_to_grid(length)
    p = gf.path.straight(length=length, npoints=npoints)
    x = gf.get_cross_section(cross_section, **kwargs)

    c = Component()
    path = p.extrude(x)
    ref = c << path
    c.add_ports(ref.ports)
    c.info["length"] = length
    c.info["width"] = x.width
    c.info["cross_section"] = cross_section

    if x.info:
        c.info.update(x.info)

    if with_bbox and length:
        padding = []
        for offset in x.bbox_offsets:
            points = get_padding_points(
                component=c,
                default=0,
                bottom=offset,
                top=offset,
            )
            padding.append(points)

        for layer, points in zip(x.bbox_layers, padding):
            c.add_polygon(points, layer=layer)
    c.absorb(ref)
    return c


if __name__ == "__main__":
    # c = straight(cross_section=gf.partial('metal_routing', width=2), with_hash=False)
    # c = straight(
    #     cross_section=gf.partial(gf.cross_section.strip, width=2), with_hash=False
    # )
    # c = straight(cladding_offset=2.5)

    nm = 1e-3
    c = straight(width=101 * nm)
    print(c.name)

    # strip2 = gf.get_cross_section("strip", layer=(2, 0))
    # settings = dict(width=2)

    # c = straight(
    #     length=1, cross_section={"cross_section": "strip", "settings": settings}
    # )
    # c = straight(
    #     length=1,
    #     cross_section={"cross_section": "strip", "settings": settings},
    #     width=3,
    #     # bbox_layers=[(2, 0)],
    #     # bbox_offsets=[3],
    # )
    # c.assert_ports_on_grid()
    c.show(show_ports=True)
    # c.plot()
    # c.pprint()

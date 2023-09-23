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
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
        cross_section: specification (CrossSection, string or dict).

    .. code::

        o1 -------------- o2
                length
    """
    p = gf.path.straight(length=length, npoints=npoints)
    x = gf.get_cross_section(cross_section)

    c = Component()
    path = p.extrude(x)
    ref = c << path
    c.add_ports(ref.ports)
    c.info["length"] = length
    c.info["width"] = x.sections[0].width
    c.info["cross_section"] = cross_section
    c.absorb(ref)
    return c


if __name__ == "__main__":
    # c = straight(cross_section=partial('metal_routing', width=2), with_hash=False)
    # c = straight(
    #     cross_section=partial(gf.cross_section.strip, width=2), with_hash=False
    # )
    # c = straight(cladding_offset=2.5)

    # nm = 1e-3
    # xs = gf.cross_section.strip()
    c = straight()
    # print(c.settings)
    # print(c.settings.info["settings"]["add_pins"])

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

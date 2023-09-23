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
    import gdsfactory as gf

    xs = gf.cross_section.strip()
    c = straight(cross_section=xs)
    c.show(show_ports=True)
"xs_sc" "xs_sc" "xs_sc"

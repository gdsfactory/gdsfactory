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
    add_pins: bool = True,
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string or dict).
        add_pins: add pins to the component.

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
    if add_pins:
        c = x.add_pins(c)
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

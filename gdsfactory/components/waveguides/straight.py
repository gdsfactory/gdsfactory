"""Straight waveguide."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.typings import CrossSectionSpec


@gf.cell_with_module_name
def straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.

    .. code::

        o1  ──────────────── o2
                length
    """
    if width is not None:
        x = gf.get_cross_section(cross_section, width=width)
    else:
        x = gf.get_cross_section(cross_section)
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
    width: float | None = None,
) -> ComponentAllAngle:
    """Returns a Straight waveguide with offgrid ports.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.

    .. code::

        o1  ──────────────── o2
                length
    """
    if width is not None:
        x = gf.get_cross_section(cross_section, width=width)
    else:
        x = gf.get_cross_section(cross_section)
    p = gf.path.straight(length=length, npoints=npoints)
    c = p.extrude(x, all_angle=True)
    x.add_bbox(c)

    c.info["length"] = length
    c.info["width"] = x.width if len(x.sections) == 0 else x.sections[0].width
    c.add_route_info(cross_section=x, length=length)
    return c


@gf.cell_with_module_name
def straight_array(
    n: int = 4,
    spacing: float = 4.0,
    length: float = 10.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Array of straights connected with grating couplers.

    useful to align the 4 corners of the chip

    Args:
        n: number of straights.
        spacing: edge to edge straight spacing.
        length: straight length (um).
        cross_section: specification (CrossSection, string or dict).
    """
    c = Component()
    wg = straight(cross_section=cross_section, length=length)

    for i in range(n):
        wref = c.add_ref(wg)
        wref.y += i * (spacing + wg.info["width"])
        c.add_ports(wref.ports, prefix=str(i))

    c.auto_rename_ports()
    return c


@gf.cell_with_module_name
def wire_straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "metal_routing",
    width: float | None = None,
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.

    .. code::

        o1  ──────────────── o2
                length
    """
    if width is not None:
        x = gf.get_cross_section(cross_section, width=width)
    else:
        x = gf.get_cross_section(cross_section)
    p = gf.path.straight(length=length, npoints=npoints)
    c = p.extrude(x)
    x.add_bbox(c)

    c.info["length"] = length
    c.info["width"] = x.width if len(x.sections) == 0 else x.sections[0].width
    c.add_route_info(cross_section=x, length=length)
    return c


if __name__ == "__main__":
    c = straight(width=10)
    print(c.ports)

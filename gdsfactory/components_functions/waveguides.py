"""Waveguide component functions without decorators."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.typings import CrossSectionSpec


def straight_function(
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


def straight_all_angle_function(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
) -> ComponentAllAngle:
    """Returns a Straight all angle waveguide.

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

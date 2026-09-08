"""Straight waveguide."""

from __future__ import annotations

__all__ = ["straight", "straight_all_angle", "straight_array", "wire_straight"]

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.typings import CrossSectionSpec

from .._schematic import straight_schematic, wire_schematic


@gf.cell_with_module_name(schematic_function=straight_schematic, tags=["waveguides"])
def straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
    port_names: tuple[str | None, str | None] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "optical"),
    **kwargs: Any,
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: native cross-section specification, string or dict.
        width: width of the waveguide. If None, it will use the width of the cross_section.
        port_names: names for the two ports.
        port_types: types for the two ports.
        kwargs: keyword-only options, including ``port_cross_section``; if False,
            omit native cross-section metadata from ports.

        o1  ──────────────── o2
                length
    """
    port_cross_section = kwargs.pop("port_cross_section", True)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {tuple(kwargs)}")

    if port_names == ("o1", "o2") and port_types == ("optical", "optical"):
        metadata = gf.get_cross_section_port_metadata(cross_section)
        if metadata is not None:
            port_names, port_types = metadata
    x = gf.get_cross_section(cross_section)
    if width is not None and width != x.width:
        x = gf.cross_section.copy_cross_section(x, width=width)
    p = gf.path.straight(length=length, npoints=npoints)
    c = p.extrude(
        x,
        port_names=port_names,
        port_types=port_types,
        port_cross_section=port_cross_section,
        add_bbox=True,
    )

    c.info["length"] = length
    c.info["width"] = x.width
    c.add_route_info(cross_section=x, length=length)
    return c


@gf.vcell
def straight_all_angle(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
    port_names: tuple[str | None, str | None] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "optical"),
) -> ComponentAllAngle:
    """Returns a Straight waveguide with offgrid ports.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: native cross-section specification, string or dict.
        width: width of the waveguide. If None, it will use the width of the cross_section.
        port_names: names for the two ports.
        port_types: types for the two ports.

        o1  ──────────────── o2
                length
    """
    if port_names == ("o1", "o2") and port_types == ("optical", "optical"):
        metadata = gf.get_cross_section_port_metadata(cross_section)
        if metadata is not None:
            port_names, port_types = metadata
    x = gf.get_cross_section(cross_section)
    if width is not None and width != x.width:
        x = gf.cross_section.copy_cross_section(x, width=width)
    p = gf.path.straight(length=length, npoints=npoints)
    c = p.extrude(
        x,
        port_names=port_names,
        port_types=port_types,
        all_angle=True,
        add_bbox=True,
    )

    c.info["length"] = length
    c.info["width"] = x.width
    c.add_route_info(cross_section=x, length=length)
    return c


@gf.cell_with_module_name(tags=["waveguides"])
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
        cross_section: native cross-section specification, string or dict.
    """
    c = Component()
    wg = straight(cross_section=cross_section, length=length)

    for i in range(n):
        wref = c.add_ref(wg)
        wref.y += i * (spacing + wg.info["width"])
        c.add_ports(wref.ports, prefix=str(i))

    c.auto_rename_ports()
    return c


@gf.cell_with_module_name(schematic_function=wire_schematic, tags=["waveguides"])
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
        cross_section: native cross-section specification, string or dict.
        width: width of the waveguide. If None, it will use the width of the cross_section.

        o1  ──────────────── o2
                length
    """
    x = gf.get_cross_section(cross_section)
    if width is not None and width != x.width:
        x = gf.cross_section.copy_cross_section(x, width=width)
    p = gf.path.straight(length=length, npoints=npoints)
    c = p.extrude(
        x,
        port_names=("e1", "e2"),
        port_types=("electrical", "electrical"),
        add_bbox=True,
    )

    c.info["length"] = length
    c.info["width"] = x.width
    c.add_route_info(cross_section=x, length=length)
    return c

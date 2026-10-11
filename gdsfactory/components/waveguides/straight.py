"""Straight waveguide."""

from __future__ import annotations

__all__ = ["straight", "straight_all_angle", "straight_array", "wire_straight"]

from typing import Unpack

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.typings import CrossSectionSpec, ExtrusionPorts

from .._schematic import straight_schematic, wire_schematic


@gf.cell_with_module_name(schematic_function=straight_schematic, tags=["waveguides"])
def straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
    **kwargs: Unpack[ExtrusionPorts],
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.
        kwargs: optional ``port_type`` override for ports o1/o2.
            Defaults to the PDK's port policy.

    ```text
        o1  ──────────────── o2
                length
    ```
    """
    return cf.straight(
        length=length,
        npoints=npoints,
        cross_section=cross_section,
        width=width,
        **kwargs,
    )


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

    ```text
        o1  ──────────────── o2
                length
    ```
    """
    return cf.straight_all_angle(
        length=length, npoints=npoints, cross_section=cross_section, width=width
    )


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
        cross_section: specification (CrossSection, string or dict).
    """
    return cf.straight_array(
        n=n, spacing=spacing, length=length, cross_section=cross_section
    )


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
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.

    ```text
        o1  ──────────────── o2
                length
    ```
    """
    return cf.wire_straight(
        length=length, npoints=npoints, cross_section=cross_section, width=width
    )

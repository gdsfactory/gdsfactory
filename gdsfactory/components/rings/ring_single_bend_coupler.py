from __future__ import annotations

__all__ = ["coupler_bend", "coupler_ring_bend", "ring_single_bend_coupler"]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import AnyComponentFactory, ComponentSpec, CrossSectionSpec

from .._schematic import (
    coupler_ring_schematic,
    coupler_schematic,
    ring_single_schematic,
)


@gf.cell_with_module_name(schematic_function=coupler_schematic, tags=["rings"])
def coupler_bend(
    radius: float | None = None,
    coupler_gap: float = 0.2,
    coupling_angle_coverage: float = 120.0,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
    bend: str | AnyComponentFactory = "bend_circular_all_angle",
    bend_output: ComponentSpec = "bend_euler",
) -> Component:
    r"""Compact curved coupler with bezier escape.

    TODO: fix for euler bends.

    Args:
        radius: um.
        coupler_gap: um.
        coupling_angle_coverage: degrees.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.
        bend: for bend.
        bend_output: for bend.

    ```text
            r   4
            |   |
            |  / ___3
            | / /
        2____/ /
        1_____/
    ```
    """
    return cf.coupler_bend(
        radius=radius,
        coupler_gap=coupler_gap,
        coupling_angle_coverage=coupling_angle_coverage,
        cross_section_inner=cross_section_inner,
        cross_section_outer=cross_section_outer,
        bend=bend,
        bend_output=bend_output,
    )


@gf.cell_with_module_name(schematic_function=coupler_ring_schematic, tags=["rings"])
def coupler_ring_bend(
    radius: float | None = None,
    coupler_gap: float = 0.2,
    coupling_angle_coverage: float = 90.0,
    length_x: float = 0.0,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
    bend: str | AnyComponentFactory = "bend_circular_all_angle",
    bend_output: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
) -> Component:
    r"""Two back-to-back coupler_bend.

    Args:
        radius: um. Default is None, which uses the default radius of the cross_section.
        coupler_gap: um.
        coupling_angle_coverage: degrees.
        length_x: horizontal straight length.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.
        bend: for bend.
        bend_output: for bend.
        straight: for straight.
    """
    return cf.coupler_ring_bend(
        radius=radius,
        coupler_gap=coupler_gap,
        coupling_angle_coverage=coupling_angle_coverage,
        length_x=length_x,
        cross_section_inner=cross_section_inner,
        cross_section_outer=cross_section_outer,
        bend=bend,
        bend_output=bend_output,
        straight=straight,
    )


@gf.cell_with_module_name(schematic_function=ring_single_schematic, tags=["rings"])
def ring_single_bend_coupler(
    radius: float = 5.0,
    gap: float = 0.2,
    coupling_angle_coverage: float = 180.0,
    bend_all_angle: str | AnyComponentFactory = "bend_circular_all_angle",
    bend: ComponentSpec = "bend_circular",
    bend_output: ComponentSpec = "bend_euler",
    length_x: float = 0.6,
    length_y: float = 0.6,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
    **kwargs: Any,
) -> Component:
    r"""Returns ring with curved coupler.

    TODO: enable euler bends.

    Args:
        radius: um.
        gap: um.
        coupling_angle_coverage: degrees.
        bend_all_angle: for bend.
        bend: for bend.
        bend_output: for bend.
        length_x: horizontal straight length.
        length_y: vertical straight length.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.
        kwargs: cross_section settings.
    """
    return cf.ring_single_bend_coupler(
        radius=radius,
        gap=gap,
        coupling_angle_coverage=coupling_angle_coverage,
        bend_all_angle=bend_all_angle,
        bend=bend,
        bend_output=bend_output,
        length_x=length_x,
        length_y=length_y,
        cross_section_inner=cross_section_inner,
        cross_section_outer=cross_section_outer,
        **kwargs,
    )

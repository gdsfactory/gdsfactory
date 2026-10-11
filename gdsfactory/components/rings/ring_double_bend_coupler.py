from __future__ import annotations

__all__ = ["ring_double_bend_coupler"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentAllAngleSpec, CrossSectionSpec

from .._schematic import ring_double_schematic


@gf.cell_with_module_name(schematic_function=ring_double_schematic, tags=["rings"])
def ring_double_bend_coupler(
    radius: float = 5.0,
    gap: float = 0.2,
    coupling_angle_coverage: float = 70.0,
    bend: ComponentAllAngleSpec = "bend_circular_all_angle",
    length_x: float = 0.6,
    length_y: float = 0.6,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
) -> Component:
    r"""Returns ring with double curved couplers.

    Args:
        radius: um.
        gap: um.
        coupling_angle_coverage: degrees.
        bend: for bend.
        length_x: horizontal straight length.
        length_y: vertical straight length.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.
    """
    return cf.ring_double_bend_coupler(
        radius=radius,
        gap=gap,
        coupling_angle_coverage=coupling_angle_coverage,
        bend=bend,
        length_x=length_x,
        length_y=length_y,
        cross_section_inner=cross_section_inner,
        cross_section_outer=cross_section_outer,
    )

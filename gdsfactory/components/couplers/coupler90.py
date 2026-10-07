from __future__ import annotations

__all__ = ["coupler90", "coupler90circular"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import coupler_schematic


@gf.cell_with_module_name(schematic_function=coupler_schematic, tags=["couplers"])
def coupler90(
    gap: float = 0.2,
    radius: float | None = None,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
    cross_section_bend: CrossSectionSpec | None = None,
    length_straight: float | None = None,
) -> Component:
    r"""Straight coupled to a bend.

    Args:
        gap: um.
        radius: um.
        straight: for straight.
        bend: bend spec.
        cross_section: cross_section spec.
        cross_section_bend: optional bend cross_section spec.
        length_straight: optional length of the straight waveguide.

    ```text
            o3
             |
            /
           /
       o2_/
       o1___o4
    ```

    """
    return cf.coupler90(
        gap=gap,
        radius=radius,
        bend=bend,
        straight=straight,
        cross_section=cross_section,
        cross_section_bend=cross_section_bend,
        length_straight=length_straight,
    )


coupler90circular = CellAlias(coupler90, bend="bend_circular")

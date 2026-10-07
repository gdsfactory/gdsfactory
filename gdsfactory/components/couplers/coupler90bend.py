from __future__ import annotations

__all__ = ["coupler90bend"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import coupler_schematic


@gf.cell_with_module_name(schematic_function=coupler_schematic, tags=["couplers"])
def coupler90bend(
    radius: float = 10.0,
    gap: float = 0.2,
    bend: ComponentSpec = "bend_euler",
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
) -> Component:
    r"""Returns 2 coupled bends.

    Args:
        radius: um.
        gap: um.
        bend: for bend.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.

    ```text
            r   3 4
            |   | |
            |  / /
            | / /
        2____/ /
        1_____/
    ```

    """
    return cf.coupler90bend(
        radius=radius,
        gap=gap,
        bend=bend,
        cross_section_inner=cross_section_inner,
        cross_section_outer=cross_section_outer,
    )

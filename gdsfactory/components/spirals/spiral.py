from __future__ import annotations

__all__ = ["spiral"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.cross_section import CrossSectionSpec
from gdsfactory.typings import ComponentSpec

from .._schematic import spiral_schematic


@gf.cell_with_module_name(schematic_function=spiral_schematic, tags=["spirals"])
def spiral(
    length: float = 100,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
    spacing: float = 3.0,
    n_loops: int = 6,
) -> gf.Component:
    """Returns a spiral double (spiral in, and then out).

    Args:
        length: length of the spiral straight section.
        bend: bend component.
        straight: straight component.
        cross_section: cross_section component.
        spacing: spacing between the spiral loops.
        n_loops: number of loops.
    """
    return cf.spiral(
        length=length,
        bend=bend,
        straight=straight,
        cross_section=cross_section,
        spacing=spacing,
        n_loops=n_loops,
    )

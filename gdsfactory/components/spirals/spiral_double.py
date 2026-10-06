from __future__ import annotations

__all__ = ["spiral_double"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import spiral_schematic


@gf.cell_with_module_name(schematic_function=spiral_schematic, tags=["spirals"])
def spiral_double(
    min_bend_radius: float = 10.0,
    separation: float = 2.0,
    number_of_loops: float = 3,
    npoints: int = 1000,
    cross_section: CrossSectionSpec = "strip",
    bend: ComponentSpec = "bend_circular",
) -> gf.Component:
    """Returns a spiral double (spiral in, and then out).

    Args:
        min_bend_radius: inner radius of the spiral.
        separation: separation between the loops.
        number_of_loops: number of loops per spiral.
        npoints: points for the spiral.
        cross_section: cross-section to extrude the structure with.
        bend: factory for the bends in the middle of the double spiral.
    """
    return cf.spiral_double(
        min_bend_radius=min_bend_radius,
        separation=separation,
        number_of_loops=number_of_loops,
        npoints=npoints,
        cross_section=cross_section,
        bend=bend,
    )

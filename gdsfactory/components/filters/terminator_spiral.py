from __future__ import annotations

__all__ = ["terminator_spiral"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.typings import CrossSectionSpec

from .._schematic import terminator_schematic


@gf.cell_with_module_name(schematic_function=terminator_schematic, tags=["filters"])
def terminator_spiral(
    separation: float = 3.0,
    width_tip: float = 0.2,
    number_of_loops: float = 1,
    npoints: int = 1000,
    min_bend_radius: float | None = None,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """Returns doped taper to terminate waveguides.

    Args:
        separation: separation between the loops.
        width_tip: width of the default cross-section at the end of the termination.
            Only used if cross_section_tip is not None.
        number_of_loops: number of loops in the spiral.
        npoints: points for the spiral.
        min_bend_radius: minimum bend radius for the spiral.
        cross_section: input cross-section.
    """
    return cf.terminator_spiral(
        separation=separation,
        width_tip=width_tip,
        number_of_loops=number_of_loops,
        npoints=npoints,
        min_bend_radius=min_bend_radius,
        cross_section=cross_section,
    )

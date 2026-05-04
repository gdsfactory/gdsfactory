from __future__ import annotations

__all__ = ["spiral_double"]

import gdsfactory as gf
from gdsfactory.path import spiral_archimedean
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
    component = gf.Component()

    bend_inner = gf.get_component(
        bend, radius=min_bend_radius / 2, angle=180, cross_section=cross_section
    )
    bend_outer = gf.get_component(
        bend, radius=min_bend_radius / 2, angle=90, cross_section=cross_section
    )

    bend1 = component.add_ref(bend_inner)
    bend2 = component.add_ref(bend_inner)
    bend2.connect("o2", bend1.ports["o1"], mirror=True)

    path = spiral_archimedean(
        min_bend_radius=min_bend_radius,
        separation=separation,
        number_of_loops=number_of_loops,
        npoints=npoints,
    )
    path.start_angle = 0
    path.end_angle = 0

    spiral = path.extrude(cross_section=cross_section)
    spiral1 = component.add_ref(spiral)
    spiral2 = component.add_ref(spiral)
    spiral2.mirror()

    spiral2.connect("o1", bend2.ports["o1"])
    spiral1.connect("o1", bend1.ports["o2"], mirror=True)

    bout1 = component.add_ref(bend_outer)
    bout1.connect("o1", spiral1.ports["o2"], mirror=True)
    bout2 = component.add_ref(bend_outer)
    bout2.connect("o1", spiral2.ports["o2"], mirror=True)

    component.add_port("o1", port=bout1.ports["o2"])
    component.add_port("o2", port=bout2.ports["o2"])
    component.info["length"] = (
        float(path.length() + bend_inner.info["length"] + bend_outer.info["length"]) * 2
    )
    component.flatten()
    return component

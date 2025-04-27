from __future__ import annotations

import gdsfactory as gf
from gdsfactory.path import extrude_transition, spiral_archimedean, transition
from gdsfactory.typings import CrossSectionSpec


@gf.cell_with_module_name
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
    cross_section_main = gf.get_cross_section(cross_section)
    cross_section_tip = gf.get_cross_section(cross_section, width=width_tip)

    xs = transition(
        cross_section2=cross_section_main,
        cross_section1=cross_section_tip,
        width_type="linear",
    )

    min_bend_radius = min_bend_radius or cross_section_main.radius_min
    assert min_bend_radius

    path = spiral_archimedean(
        min_bend_radius=min_bend_radius,
        separation=separation / 2,
        number_of_loops=number_of_loops,
        npoints=npoints,
    )
    path.start_angle = 0
    path.end_angle = 0

    spiral = extrude_transition(path, transition=xs)
    c = gf.Component()
    ref = c << spiral
    c.add_port("o1", port=ref["o1"])
    return c


if __name__ == "__main__":
    c = terminator_spiral(number_of_loops=3)
    c.show()

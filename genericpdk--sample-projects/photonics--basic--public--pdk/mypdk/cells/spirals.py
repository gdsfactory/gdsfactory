"""Rings."""

import gdsfactory as gf
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    Floats,
)


@gf.cell
def spiral(
    length: float = 100,
    cross_section: CrossSectionSpec = "strip",
    spacing: float = 3,
    n_loops: int = 6,
) -> gf.Component:
    """Returns a spiral double (spiral in, and then out).

    Args:
        length: length of the spiral straight section.
        cross_section: cross_section component.
        spacing: spacing between the spiral loops.
        n_loops: number of loops.
    """
    return gf.c.spiral(
        length=length,
        cross_section=cross_section,
        spacing=spacing,
        n_loops=n_loops,
        bend="bend_euler",
        straight="straight",
    )


@gf.cell
def spiral_racetrack(
    min_radius: float | None = None,
    straight_length: float = 20.0,
    spacings: Floats = (2, 2, 3, 3, 2, 2),
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_euler",
    bend_s: ComponentSpec = "bend_s",
    cross_section: CrossSectionSpec = "strip",
    cross_section_s: CrossSectionSpec | None = None,
    extra_90_deg_bend: bool = False,
    allow_min_radius_violation: bool = False,
) -> gf.Component:
    """Returns Racetrack-Spiral.

    Args:
        min_radius: smallest radius in um.
        straight_length: length of the straight segments in um.
        spacings: space between the center of neighboring waveguides in um.
        straight: factory to generate the straight segments.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        cross_section: cross-section of the waveguides.
        cross_section_s: cross-section of the s bend waveguide (optional).
        extra_90_deg_bend: if True, we add an additional straight + 90 degree bent at the output, so the output port is looking down.
        allow_min_radius_violation: if True, will allow the s-bend to have a smaller radius than the minimum radius.
    """
    return gf.c.spiral_racetrack(
        min_radius=min_radius,
        straight_length=straight_length,
        spacings=spacings,
        straight=straight,
        bend=bend,
        bend_s=bend_s,
        cross_section=cross_section,
        cross_section_s=cross_section_s,
        extra_90_deg_bend=extra_90_deg_bend,
        allow_min_radius_violation=allow_min_radius_violation,
    )


@gf.cell
def spiral_racetrack_heater(
    spacing: float = 4.0,
    num: int = 8,
    straight_length: float = 100,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """Returns spiral racetrack with a heater above.

    based on https://doi.org/10.1364/OL.400230 .

    Args:
        spacing: center to center spacing between the waveguides.
        num: number of spiral loops.
        straight_length: length of the straight segments.
        cross_section: cross_section.
    """
    return gf.c.spiral_racetrack_heater_metal(
        straight_length=straight_length,
        min_radius=None,
        spacing=spacing,
        num=num,
        straight="straight",
        bend="bend_euler",
        bend_s=gf.get_cell("bend_s"),
        waveguide_cross_section=cross_section,
        via_stack="via_stack_heater_mtop",
    )

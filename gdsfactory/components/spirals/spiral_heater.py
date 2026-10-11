from __future__ import annotations

__all__ = [
    "spiral_racetrack",
    "spiral_racetrack_fixed_length",
    "spiral_racetrack_heater_doped",
    "spiral_racetrack_heater_metal",
]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    Floats,
)

from .._schematic import spiral_schematic


@gf.cell_with_module_name(schematic_function=spiral_schematic, tags=["spirals"])
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
) -> Component:
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
    return cf.spiral_racetrack(
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


@gf.cell_with_module_name(schematic_function=spiral_schematic, tags=["spirals"])
def spiral_racetrack_fixed_length(
    length: float = 1000,
    in_out_port_spacing: float = 150,
    n_straight_sections: int = 8,
    min_radius: float | None = None,
    min_spacing: float = 5.0,
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_circular",
    bend_s: ComponentSpec = "bend_s",
    cross_section: CrossSectionSpec = "strip",
    cross_section_s: CrossSectionSpec | None = None,
) -> Component:
    """Returns Racetrack-Spiral with a specified total length.

    The input and output ports are aligned in y. This class is meant to
    be used for generating interferometers with long waveguide lengths, where
    the most important parameter is the length difference between the arms.

    Args:
        length: total length of the spiral from input to output ports in um.
        in_out_port_spacing: spacing between input and output ports of the spiral in um.
        n_straight_sections: total number of straight sections for the racetrack spiral. Has to be even.
        min_radius: smallest radius in um.
        min_spacing: minimum center-center spacing between adjacent waveguides.
        straight: factory to generate the straight segments.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        cross_section: cross-section of the waveguides.
        cross_section_s: cross-section of the s bend waveguide (optional).
    """
    return cf.spiral_racetrack_fixed_length(
        length=length,
        in_out_port_spacing=in_out_port_spacing,
        n_straight_sections=n_straight_sections,
        min_radius=min_radius,
        min_spacing=min_spacing,
        straight=straight,
        bend=bend,
        bend_s=bend_s,
        cross_section=cross_section,
        cross_section_s=cross_section_s,
    )


@gf.cell_with_module_name(schematic_function=spiral_schematic, tags=["spirals"])
def spiral_racetrack_heater_metal(
    min_radius: float | None = None,
    straight_length: float = 30,
    spacing: float = 2,
    num: int = 8,
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_euler",
    bend_s: ComponentSpec = "bend_s",
    waveguide_cross_section: CrossSectionSpec = "strip",
    heater_cross_section: CrossSectionSpec = "heater_metal",
    via_stack: ComponentSpec | None = "via_stack_heater_mtop",
) -> Component:
    """Returns spiral racetrack with a heater above.

    based on <https://doi.org/10.1364/OL.400230> .

    Args:
        min_radius: smallest radius. Defaults to the radius of the cross-section.
        straight_length: length of the straight segments.
        spacing: space between the center of neighboring waveguides.
        num: number of loops.
        straight: factory to generate the straight segments.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        waveguide_cross_section: cross-section of the waveguides.
        heater_cross_section: cross-section of the heater.
        via_stack: via stack to connect the heater to the metal layer.
    """
    return cf.spiral_racetrack_heater_metal(
        min_radius=min_radius,
        straight_length=straight_length,
        spacing=spacing,
        num=num,
        straight=straight,
        bend=bend,
        bend_s=bend_s,
        waveguide_cross_section=waveguide_cross_section,
        heater_cross_section=heater_cross_section,
        via_stack=via_stack,
    )


@gf.cell_with_module_name(schematic_function=spiral_schematic, tags=["spirals"])
def spiral_racetrack_heater_doped(
    min_radius: float | None = None,
    straight_length: float = 30,
    spacing: float = 2,
    num: int = 8,
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_euler",
    bend_s: ComponentSpec = "bend_s",
    waveguide_cross_section: CrossSectionSpec = "strip",
    heater_cross_section: CrossSectionSpec = "npp",
) -> Component:
    """Returns spiral racetrack with a heater between the loops.

    based on <https://doi.org/10.1364/OL.400230> but with the heater between the loops.

    Args:
        min_radius: smallest radius in um. Defaults to the radius of the cross-section.
        straight_length: length of the straight segments in um.
        spacing: space between the center of neighboring waveguides in um.
        num: number.
        straight: factory to generate the straight segments.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        waveguide_cross_section: cross-section of the waveguides.
        heater_cross_section: cross-section of the heater.
    """
    return cf.spiral_racetrack_heater_doped(
        min_radius=min_radius,
        straight_length=straight_length,
        spacing=spacing,
        num=num,
        straight=straight,
        bend=bend,
        bend_s=bend_s,
        waveguide_cross_section=waveguide_cross_section,
        heater_cross_section=heater_cross_section,
    )

from __future__ import annotations

__all__ = [
    "straight_heater_metal",
    "straight_heater_metal_90_90",
    "straight_heater_metal_simple",
    "straight_heater_metal_undercut",
    "straight_heater_metal_undercut_90_90",
]

from functools import partial

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import straight_schematic


@gf.cell_with_module_name(schematic_function=straight_schematic, tags=["waveguides"])
def straight_heater_metal_undercut(
    length: float = 320.0,
    length_undercut_spacing: float = 6.0,
    length_undercut: float = 30.0,
    length_straight: float = 0.1,
    length_straight_input: float = 15.0,
    cross_section: CrossSectionSpec = "strip",
    cross_section_heater: CrossSectionSpec = "heater_metal",
    cross_section_waveguide_heater: CrossSectionSpec = "strip_heater_metal",
    cross_section_heater_undercut: CrossSectionSpec = "strip_heater_metal_undercut",
    with_undercut: bool = True,
    via_stack: ComponentSpec | None = "via_stack_heater_mtop",
    port_orientation1: int | None = None,
    port_orientation2: int | None = None,
    heater_taper_length: float = 5.0,
    ohms_per_square: float | None = None,
) -> Component:
    """Returns a thermal phase shifter.

    dimensions from <https://doi.org/10.1364/OE.27.010456>

    Args:
        length: of the waveguide.
        length_undercut_spacing: from undercut regions.
        length_undercut: length of each undercut section.
        length_straight: length of the straight waveguide.
        length_straight_input: from input port to where trenches start.
        cross_section: for waveguide ports.
        cross_section_heater: for heated sections. heater metal only.
        cross_section_waveguide_heater: for heated sections.
        cross_section_heater_undercut: for heated sections with undercut.
        with_undercut: isolation trenches for higher efficiency.
        via_stack: via stack.
        port_orientation1: left via stack port orientation. None adds all orientations.
        port_orientation2: right via stack port orientation. None adds all orientations.
        heater_taper_length: minimizes current concentrations from heater to via_stack.
        ohms_per_square: to calculate resistance.
    """
    return cf.straight_heater_metal_undercut(
        length=length,
        length_undercut_spacing=length_undercut_spacing,
        length_undercut=length_undercut,
        length_straight=length_straight,
        length_straight_input=length_straight_input,
        cross_section=cross_section,
        cross_section_heater=cross_section_heater,
        cross_section_waveguide_heater=cross_section_waveguide_heater,
        cross_section_heater_undercut=cross_section_heater_undercut,
        with_undercut=with_undercut,
        via_stack=via_stack,
        port_orientation1=port_orientation1,
        port_orientation2=port_orientation2,
        heater_taper_length=heater_taper_length,
        ohms_per_square=ohms_per_square,
    )


@gf.cell_with_module_name(schematic_function=straight_schematic, tags=["waveguides"])
def straight_heater_metal_simple(
    length: float = 320.0,
    cross_section_heater: CrossSectionSpec = "heater_metal",
    cross_section_waveguide_heater: CrossSectionSpec = "strip_heater_metal",
    via_stack: ComponentSpec | None = "via_stack_heater_mtop",
    port_orientation1: int | None = None,
    port_orientation2: int | None = None,
    heater_taper_length: float = 5.0,
    ohms_per_square: float | None = None,
) -> Component:
    """Returns a thermal phase shifter that has properly fixed electrical connectivity to extract a suitable electrical netlist and models.

    dimensions from <https://doi.org/10.1364/OE.27.010456>.

    Args:
        length: of the waveguide.
        cross_section_heater: for heated sections. heater metal only.
        cross_section_waveguide_heater: for heated sections.
        via_stack: via stack.
        port_orientation1: left via stack port orientation. None adds all orientations.
        port_orientation2: right via stack port orientation. None adds all orientations.
        heater_taper_length: minimizes current concentrations from heater to via_stack.
        ohms_per_square: to calculate resistance.
    """
    return cf.straight_heater_metal_simple(
        length=length,
        cross_section_heater=cross_section_heater,
        cross_section_waveguide_heater=cross_section_waveguide_heater,
        via_stack=via_stack,
        port_orientation1=port_orientation1,
        port_orientation2=port_orientation2,
        heater_taper_length=heater_taper_length,
        ohms_per_square=ohms_per_square,
    )


straight_heater_metal = partial(
    straight_heater_metal_undercut,
    with_undercut=False,
    length_straight_input=0.1,
    length_undercut=5,
    length_undercut_spacing=0,
)
straight_heater_metal_90_90 = partial(
    straight_heater_metal,
    port_orientation1=90,
    port_orientation2=90,
)
straight_heater_metal_undercut_90_90 = partial(
    straight_heater_metal_undercut,
    port_orientation1=90,
    port_orientation2=90,
)

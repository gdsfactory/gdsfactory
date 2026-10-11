"""Straight Ge photodetector."""

from __future__ import annotations

__all__ = ["ge_detector_straight_si_contacts"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import photodiode_schematic


@gf.cell_with_module_name(schematic_function=photodiode_schematic, tags=["detectors"])
def ge_detector_straight_si_contacts(
    length: float = 40.0,
    cross_section: CrossSectionSpec = "pn_ge_detector_si_contacts",
    via_stack: ComponentSpec = "via_stack_slab_m3",
    via_stack_width: float = 10.0,
    via_stack_spacing: float = 5.0,
    via_stack_offset: float = 0.0,
    taper_length: float = 20.0,
    taper_width: float = 0.8,
    taper_cros_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns a straight Ge on Si detector with silicon contacts.

    There are no contacts on the Ge. These detectors could have lower
    dark current and sensitivity compared to those with contacts in the
    Ge. See Chen et al., "High-Responsivity Low-Voltage 28-Gb/s Ge p-i-n
    Photodetector With Silicon Contacts", Journal of Lightwave Technology 33(4), 2015.

    <https://doi.org/10.1109/JLT.2014.2367134>

    Args:
        length: pd length.
        cross_section: for the waveguide.
        via_stack: for the via_stacks. First element
        via_stack_width: width of the via_stack.
        via_stack_spacing: spacing between via_stacks.
        via_stack_offset: with respect to the detector
        taper_length: length of the taper.
        taper_width: width of the taper.
        taper_cros_section: cross_section of the taper.
    """
    return cf.ge_detector_straight_si_contacts(
        length=length,
        cross_section=cross_section,
        via_stack=via_stack,
        via_stack_width=via_stack_width,
        via_stack_spacing=via_stack_spacing,
        via_stack_offset=via_stack_offset,
        taper_length=taper_length,
        taper_width=taper_width,
        taper_cros_section=taper_cros_section,
    )

from __future__ import annotations

__all__ = ["disk", "disk_heater"]

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory import component_functions as cf
from gdsfactory.typings import (
    AngleInDegrees,
    ComponentSpec,
    CrossSectionSpec,
    LayerSpec,
)

from .._schematic import ring_single_schematic


@gf.cell_with_module_name(schematic_function=ring_single_schematic, tags=["rings"])
def disk(
    radius: float = 10.0,
    gap: float = 0.2,
    wrap_angle_deg: float = 180.0,
    parity: int = 1,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Disk Resonator.

    Args:
        radius: disk resonator radius.
        gap: Distance between the bus straight and resonator.
        wrap_angle_deg: Angle in degrees between 0 and 180.
            determines how much the bus straight wraps along the resonator.
            0 corresponds to a straight bus straight.
            180 corresponds to a bus straight wrapped around half of the resonator.
        parity: 1 or -1. 1 places the resonator left from the bus straight,
            -1 places it to the right.
        cross_section: cross_section spec.
    """
    return cf.disk(
        radius=radius,
        gap=gap,
        wrap_angle_deg=wrap_angle_deg,
        parity=parity,
        cross_section=cross_section,
    )


@gf.cell_with_module_name(schematic_function=ring_single_schematic, tags=["rings"])
def disk_heater(
    radius: float = 10.0,
    gap: float = 0.2,
    wrap_angle_deg: float = 180.0,
    parity: int = 1,
    cross_section: CrossSectionSpec = "strip",
    heater_layer: LayerSpec = "HEATER",
    via_stack: ComponentSpec = "via_stack_heater_mtop",
    heater_width: float = 5.0,
    heater_extent: float = 2.0,
    via_width: float = 10.0,
    port_orientation: AngleInDegrees | None = 90,
) -> Component:
    """Disk Resonator with top metal heater.

    Args:
        radius: disk resonator radius.
        gap: Distance between the bus straight and resonator.
        wrap_angle_deg: Angle in degrees between 0 and 180.
            determines how much the bus straight wraps along the resonator.
            0 corresponds to a straight bus straight.
            180 corresponds to a bus straight wrapped around half of the resonator.
        parity: 1 or -1. 1 places the resonator left from the bus straight,
            -1 places it to the right.
        cross_section: cross_section spec.
        heater_layer: layer of the heater.
        via_stack: via stack component.
        heater_width: width of the heater.
        heater_extent: length of heater beyond disk.
        via_width: size of the square via at the end of the heater.
        port_orientation: in degrees.
    """
    return cf.disk_heater(
        radius=radius,
        gap=gap,
        wrap_angle_deg=wrap_angle_deg,
        parity=parity,
        cross_section=cross_section,
        heater_layer=heater_layer,
        via_stack=via_stack,
        heater_width=heater_width,
        heater_extent=heater_extent,
        via_width=via_width,
        port_orientation=port_orientation,
    )

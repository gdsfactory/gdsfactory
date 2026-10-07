from __future__ import annotations

__all__ = ["bend_circular_heater"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from .._schematic import bend_schematic


@gf.cell_with_module_name(schematic_function=bend_schematic, tags=["bends"])
def bend_circular_heater(
    radius: float | None = None,
    angle: float = 90,
    npoints: int | None = None,
    heater_to_wg_distance: float = 1.2,
    heater_width: float = 0.5,
    layer_heater: LayerSpec = "HEATER",
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
) -> Component:
    """Creates an arc of arclength `theta` starting at angle `start_angle`.

    Args:
        radius: in um. Defaults to cross_section.radius.
        angle: angle of arc (degrees).
        npoints: Number of points used per 360 degrees.
        heater_to_wg_distance: in um.
        heater_width: in um.
        layer_heater: for heater.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
    """
    return cf.bend_circular_heater(
        radius=radius,
        angle=angle,
        npoints=npoints,
        heater_to_wg_distance=heater_to_wg_distance,
        heater_width=heater_width,
        layer_heater=layer_heater,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )

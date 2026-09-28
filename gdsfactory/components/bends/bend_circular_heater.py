from __future__ import annotations

from gdsfactory.cross_section.utils import validate_radius

__all__ = ["bend_circular_heater"]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.path import arc
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
    x = gf.get_cross_section(cross_section)
    radius = radius or x.radius
    assert radius is not None
    width = x.width

    offset = heater_to_wg_distance + width / 2
    s1 = (layer_heater, offset - heater_width / 2, offset + heater_width / 2)
    s2 = (layer_heater, -offset - heater_width / 2, -offset + heater_width / 2)
    sections = list(x.get_sections()) + [s1, s2]

    xs = gf.cross_section.cross_section(
        width=None, sections=sections, radius=x.radius, radius_min=x.radius_min
    )
    p = arc(radius=radius, angle=angle, npoints=npoints)

    c = Component()
    path = p.extrude(xs, ports={0: ("o1", "o2", "optical")})
    ref = c << path
    c.add_ports(ref.ports)
    c.info["length"] = p.length()
    c.info["dx"] = float(abs(p.points[0][0] - p.points[-1][0]))
    c.info["dy"] = float(abs(p.points[0][0] - p.points[-1][0]))
    if not allow_min_radius_violation:
        validate_radius(x, radius)
    c.flatten()
    return c

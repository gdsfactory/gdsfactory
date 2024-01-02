from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.path import arc
from gdsfactory.typings import CrossSectionSpec, LayerSpec


@gf.cell
def bend_circular_heater(
    radius: float | None = None,
    angle: float = 90,
    npoints: int | None = None,
    heater_to_wg_distance: float = 1.2,
    heater_width: float = 0.5,
    layer_heater: LayerSpec = "HEATER",
    with_bbox: bool = True,
    cross_section: CrossSectionSpec = "xs_sc",
) -> Component:
    """Creates an arc of arclength `theta` starting at angle `start_angle`.

    Args:
        radius: in um. Defaults to cross_section.radius.
        angle: angle of arc (degrees).
        npoints: Number of points used per 360 degrees.
        heater_to_wg_distance: in um.
        heater_width: in um.
        layer_heater: for heater.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        kwargs: cross_section settings.
    """
    x = gf.get_cross_section(cross_section)
    radius = radius or x.radius
    width = x.width

    offset = heater_to_wg_distance + width / 2
    s1 = gf.Section(
        width=heater_width,
        offset=+offset,
        layer=layer_heater,
    )
    s2 = gf.Section(
        width=heater_width,
        offset=-offset,
        layer=layer_heater,
    )
    sections = list(x.sections) + [s1, s2]

    xs = x.copy(sections=sections)
    p = arc(radius=radius, angle=angle, npoints=npoints)

    c = Component()
    path = p.extrude(xs)
    ref = c << path
    c.add_ports(ref.ports)
    c.absorb(ref)
    c.info["length"] = float(np.round(p.length(), 3))
    c.info["dx"] = c.info["dy"] = float(abs(p.points[0][0] - p.points[-1][0]))

    x.validate_radius(radius)
    if with_bbox:
        top = None if int(angle) in {180, -180, -90} else 0
        bottom = 0 if int(angle) in {-90} else None
        x.add_bbox(c, top=top, bottom=bottom)
    return c


if __name__ == "__main__":
    c = bend_circular_heater(heater_width=1, cross_section="xs_rc")
    print(c.ports)
    c.show(show_ports=True)

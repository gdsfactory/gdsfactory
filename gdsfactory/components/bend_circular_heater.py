from __future__ import annotations

import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.component import Component
from gdsfactory.path import arc
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import CrossSectionSpec, LayerSpec


@gf.cell
def bend_circular_heater(
    radius: float = 10,
    angle: float = 90,
    npoints: int = 720,
    heater_to_wg_distance: float = 1.2,
    heater_width: float = 0.5,
    layer_heater: LayerSpec = "HEATER",
    with_bbox: bool = True,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    """Creates an arc of arclength `theta` starting at angle `start_angle`.

    Args:
        radius: in um.
        angle: angle of arc (degrees).
        npoints: Number of points used per 360 degrees.
        heater_to_wg_distance: in um.
        heater_width: in um.
        layer_heater: for heater.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        kwargs: cross_section settings.
    """
    x = gf.get_cross_section(cross_section, radius=radius, **kwargs)
    width = x.width
    layer = x.layer

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
    x = gf.CrossSection(
        width=width, offset=0, layer=layer, port_names=["in", "out"], sections=[s1, s2]
    )

    p = arc(radius=radius, angle=angle, npoints=npoints)
    c = p.extrude(x)
    c.length = snap_to_grid(p.length())
    c.dx = abs(p.points[0][0] - p.points[-1][0])
    c.dy = abs(p.points[0][0] - p.points[-1][0])

    if with_bbox:
        padding = []
        for offset in x.bbox_offsets:
            top = offset if angle == 180 else 0
            points = get_padding_points(
                component=c,
                default=0,
                bottom=offset,
                right=offset,
                top=top,
            )
            padding.append(points)

        for layer, points in zip(x.bbox_layers, padding):
            c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    c = bend_circular_heater(heater_width=1)
    print(c.ports)
    c.show(show_ports=True)

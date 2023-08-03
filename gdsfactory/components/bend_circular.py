from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.component import Component
from gdsfactory.path import arc
from gdsfactory.route_info import route_info_from_cs
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import CrossSectionSpec


@gf.cell
def bend_circular(
    angle: float = 90.0,
    npoints: int | None = None,
    with_bbox: bool = True,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    """Returns a radial arc.

    Args:
        angle: angle of arc (degrees).
        npoints: number of points.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
        cross_section: spec (CrossSection, string or dict).
        kwargs: cross_section settings.

    .. code::

                  o2
                  |
                 /
                /
               /
       o1_____/
    """
    x = gf.get_cross_section(cross_section, **kwargs)
    radius = x.radius

    p = arc(radius=radius, angle=angle, npoints=npoints)
    c = Component()
    path = p.extrude(x)
    ref = c << path
    c.add_ports(ref.ports)

    c.absorb(ref)
    c.info["length"] = float(snap_to_grid(p.length()))
    c.info["dy"] = snap_to_grid(float(abs(p.points[0][0] - p.points[-1][0])))
    c.info["radius"] = float(radius)
    c.info["route_info"] = route_info_from_cs(
        cross_section, length=c.info["length"], n_bend_90=abs(angle / 90.0)
    )

    if with_bbox and x.bbox_layers:
        padding = []
        for offset in x.bbox_offsets:
            top = offset if angle in {180, -180, -90} else 0
            bottom = 0 if angle in {-90} else offset
            points = get_padding_points(
                component=c,
                default=0,
                bottom=bottom,
                right=offset,
                top=top,
            )
            padding.append(points)

        for layer, points in zip(x.bbox_layers, padding):
            c.add_polygon(points, layer=layer)

    return c


bend_circular180 = partial(bend_circular, angle=180)


if __name__ == "__main__":
    from gdsfactory.generic_tech import get_generic_pdk

    PDK = get_generic_pdk()
    PDK.activate()

    c = bend_circular(
        width=2,
        layer=(0, 0),
        angle=90,
        # cross_section="rib",
        with_bbox=True,
        radius=50,
    )
    # c = bend_circular()
    # c = bend_circular(cross_section=gf.cross_section.pin, radius=5)
    # c.pprint_ports()
    print(c.ports["o2"].orientation)
    c.show(show_ports=True)

    # c = bend_circular180()
    # c.plot("qt")

    # from gdsfactory.quickplotter import quickplot2
    # c = bend_circular_trenches()
    # c = bend_circular_deep_rib()
    # print(c.ports)
    # print(c.length, np.pi * 10)
    # print(c.ports.keys())
    # print(c.ports['o2'].center)
    # print(c.settings)
    # c = bend_circular_slot()
    # c = bend_circular(width=0.45, radius=5)
    # c.plot()
    # quickplot2(c)

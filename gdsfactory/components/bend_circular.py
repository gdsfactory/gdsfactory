import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.component import Component
from gdsfactory.path import arc, extrude
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import CrossSectionSpec


@gf.cell
def bend_circular(
    angle: float = 90.0,
    npoints: int = 720,
    with_bbox: bool = True,
    cross_section: CrossSectionSpec = "strip",
    **kwargs
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
    path = extrude(p, x)
    ref = c << path
    c.add_ports(ref.ports)

    c.absorb(ref)
    c.info["length"] = float(snap_to_grid(p.length()))
    c.info["dy"] = snap_to_grid(float(abs(p.points[0][0] - p.points[-1][0])))
    c.info["radius"] = float(radius)

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


bend_circular180 = gf.partial(bend_circular, angle=180)


if __name__ == "__main__":
    c = bend_circular(
        width=2, layer=(0, 0), angle=90, cross_section="rib", with_bbox=True
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

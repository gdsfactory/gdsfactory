import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.component import Component
from gdsfactory.cross_section import strip
from gdsfactory.path import arc, extrude
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import CrossSectionOrFactory


@gf.cell
def bend_circular(
    angle: int = 90,
    npoints: int = 720,
    with_cladding_box: bool = True,
    cross_section: CrossSectionOrFactory = strip,
    **kwargs
) -> Component:
    """Returns a radial arc.

    Args:
        angle: angle of arc (degrees)
        npoints: number of points
        with_cladding_box: square in layers_cladding to remove DRC
        cross_section:
        kwargs: cross_section settings


    .. code::

                  o2
                  |
                 /
                /
               /
       o1_____/

    """
    x = cross_section(**kwargs) if callable(cross_section) else cross_section
    radius = x.info["radius"]

    p = arc(radius=radius, angle=angle, npoints=npoints)
    c = Component()
    path = extrude(p, x)
    ref = c << path
    c.add_ports(ref.ports)

    c.info.length = snap_to_grid(p.length())
    c.info.dy = float(abs(p.points[0][0] - p.points[-1][0]))
    c.info.radius = float(radius)

    if with_cladding_box and x.info["layers_cladding"]:
        layers_cladding = x.info["layers_cladding"]
        cladding_offset = x.info["cladding_offset"]
        top = cladding_offset if angle == 180 else 0
        points = get_padding_points(
            component=c,
            default=0,
            bottom=cladding_offset,
            right=cladding_offset,
            top=top,
        )
        for layer in layers_cladding or []:
            c.add_polygon(points, layer=layer)

    c.absorb(ref)
    return c


bend_circular180 = gf.partial(bend_circular, angle=180)


if __name__ == "__main__":
    c = bend_circular(width=2, layer=gf.LAYER.M1)
    # c = bend_circular(cross_section=gf.cross_section.pin, radius=5)
    c.pprint()
    c.show()

    c = bend_circular180()
    c.plot("qt")

    # from phidl.quickplotter import quickplot2
    # c = bend_circular_trenches()
    # c = bend_circular_deep_rib()
    # print(c.ports)
    # print(c.length, np.pi * 10)
    # print(c.ports.keys())
    # print(c.ports['o2'].midpoint)
    # print(c.settings)
    # c = bend_circular_slot()
    # c = bend_circular(width=0.45, radius=5)
    # c.plot()
    # quickplot2(c)

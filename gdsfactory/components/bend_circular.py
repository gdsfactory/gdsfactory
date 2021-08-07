import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.component import Component
from gdsfactory.cross_section import StrOrDict, get_cross_section
from gdsfactory.path import arc, extrude
from gdsfactory.snap import snap_to_grid


@gf.cell
def bend_circular(
    angle: int = 90,
    npoints: int = 720,
    with_cladding_box: bool = True,
    waveguide: StrOrDict = "strip",
    **kwargs
) -> Component:
    """Returns a radial arc.

    Args:
        angle: angle of arc (degrees)
        npoints: number of points
        with_cladding_box: square in layers_cladding to remove DRC
        waveguide: from tech.waveguide
        kwargs: waveguide_settings

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.components.bend_circular(radius=10, angle=90, npoints=720)
        c.plot()

    """
    x = get_cross_section(waveguide, **kwargs)
    radius = x.info["radius"]

    p = arc(radius=radius, angle=angle, npoints=npoints)
    c = extrude(p, x)

    c.length = snap_to_grid(p.length())
    c.dy = abs(p.points[0][0] - p.points[-1][0])
    c.radius_min = radius
    c.waveguide_settings = x.info

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
    return c


@gf.cell
def bend_circular180(angle: int = 180, **kwargs) -> Component:
    """Returns a 180 degrees radial arc.

    Args:
        angle: angle of arc (degrees)
        npoints: number of points
        with_cladding_box: square in layers_cladding to remove DRC
        waveguide: from tech.waveguide
        kwargs: waveguide_settings

    """
    return bend_circular(angle=angle, **kwargs)


if __name__ == "__main__":
    from pprint import pprint

    # c = bend_circular(width=2, layer=gf.LAYER.M1)
    c = bend_circular(waveguide="metal_routing", widtha=3)
    c.show()
    pprint(c.get_settings())

    # c = bend_circular180()
    # c.plotqt()

    # from phidl.quickplotter import quickplot2
    # c = bend_circular_trenches()
    # c = bend_circular_deep_rib()
    # print(c.ports)
    # print(c.length, np.pi * 10)
    # print(c.ports.keys())
    # print(c.ports["N0"].midpoint)
    # print(c.settings)
    # c = bend_circular_slot()
    # c = bend_circular(width=0.45, radius=5)
    # c.plot()
    # quickplot2(c)

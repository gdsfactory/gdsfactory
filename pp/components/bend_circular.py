import pp
from pp.add_padding import get_padding_points
from pp.component import Component
from pp.cross_section import cross_section, get_waveguide_settings
from pp.path import arc, extrude
from pp.snap import snap_to_grid


@pp.cell_with_validator
def bend_circular(
    angle: int = 90,
    npoints: int = 720,
    with_cladding_box: bool = True,
    waveguide: str = "strip",
    **kwargs
) -> Component:
    """Returns a radial arc.

    Args:
        radius
        angle: angle of arc (degrees)
        with_cladding_box: to avoid DRC acute angle errors in cladding
        waveguide: from tech.waveguide
        kwargs: waveguide_settings

    .. plot::
        :include-source:

        import pp

        c = pp.components.bend_circular(radius=10, angle=90, npoints=720)
        c.plot()

    """
    waveguide_settings = get_waveguide_settings(waveguide, **kwargs)
    x = cross_section(**waveguide_settings)
    radius = x.info["radius"]

    p = arc(radius=radius, angle=angle, npoints=npoints)
    c = extrude(p, x)

    c.length = snap_to_grid(p.length())
    c.dy = abs(p.points[0][0] - p.points[-1][0])
    c.radius_min = radius

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


@pp.cell_with_validator
def bend_circular180(angle: int = 180, **kwargs) -> Component:
    """Returns a 180 degrees radial arc.

    Args:
        radius
        angle: angle of arc (degrees)
        npoints: number of points
        width: waveguide width (defaults to tech.wg_width)

    """
    return bend_circular(angle=angle, **kwargs)


if __name__ == "__main__":
    from pprint import pprint

    # c = bend_circular(width=2, layer=pp.LAYER.M1)
    c = bend_circular(waveguide="metal_routing", width=3)
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

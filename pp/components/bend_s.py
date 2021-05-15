from pp.add_padding import get_padding_points
from pp.cell import cell
from pp.component import Component
from pp.components.bezier import bezier
from pp.cross_section import cross_section, get_waveguide_settings


@cell
def bend_s(
    height: float = 2.0,
    length: float = 10.0,
    nb_points: int = 99,
    with_cladding_box: bool = True,
    waveguide: str = "strip",
    **kwargs,
) -> Component:
    """S bend with bezier curve

    Args:
        height: in y direction
        length: in x direction
        layer: gds number
        nb_points: number of points
        waveguide: from TECH.waveguide
        kwargs: waveguide_settings

    .. plot::
      :include-source:

      import pp

      c = pp.components.bend_s(height=20)
      c.plot()

    """
    l, h = length, height
    waveguide_settings = get_waveguide_settings(waveguide, **kwargs)
    x = cross_section(**waveguide_settings)
    width = x.info["width"]
    layer = x.info["layer"]

    c = bezier(
        width=width,
        control_points=[(0, 0), (l / 2, 0), (l / 2, h), (l, h)],
        npoints=nb_points,
        layer=layer,
    )
    c.add_port(name="W0", port=c.ports.pop("0"))
    c.add_port(name="E0", port=c.ports.pop("1"))

    if with_cladding_box and x.info["layers_cladding"]:
        layers_cladding = x.info["layers_cladding"]
        cladding_offset = x.info["cladding_offset"]
        points = get_padding_points(
            component=c,
            default=0,
            bottom=cladding_offset,
            top=cladding_offset,
        )
        for layer in layers_cladding or []:
            c.add_polygon(points, layer=layer)

    return c


if __name__ == "__main__":
    c = bend_s(width=1)
    c.pprint()
    # c = bend_s_biased()
    # print(c.info["min_bend_radius"])
    c.show()

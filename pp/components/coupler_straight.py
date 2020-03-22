from pp import Component
import pp


__version__ = "0.0.1"


@pp.autoname
def coupler_straight(
    length=10,
    width=0.5,
    gap=0.27,
    layer=pp.LAYER.WG,
    layers_cladding=[pp.LAYER.WGCLAD],
    cladding_offset=3,
):
    """ straight coupled waveguides. Two multimode ports

    .. plot::
      :include-source:

      import pp

      c = pp.c.coupler_straight()
      pp.plotgds(c)

    """

    c = Component()

    # Top path
    c.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)
    y = width + gap

    # Bottom path
    c.add_polygon(
        [(0, y), (length, y), (length, width + y), (0, width + y)], layer=layer
    )

    # One multimode port on each side
    port_w = width * 2 + gap

    c.add_port(name="W0", midpoint=[0, port_w / 2], width=port_w, orientation=180)
    c.add_port(name="E0", midpoint=[length, port_w / 2], width=port_w, orientation=0)

    c.width = width
    c.length = length

    # cladding
    ymax = 2 * width + gap + cladding_offset
    for layer_cladding in layers_cladding:
        c.add_polygon(
            [
                (0, -cladding_offset),
                (length, -cladding_offset),
                (length, ymax),
                (0, ymax),
            ],
            layer=layer_cladding,
        )

    return c


@pp.autoname
def coupler_straight_biased(length=10, width=0.5, gap=0.27, layer=pp.LAYER.WG):
    return coupler_straight(
        width=pp.bias.width(width), gap=pp.bias.gap(gap), length=length, layer=layer
    )


def _demo():
    c = coupler_straight(gap=0.2)
    pp.write_gds(c)
    return c


if __name__ == "__main__":
    # c = _demo()
    c = coupler_straight_biased(width=0.5, gap=0.2)
    pp.show(c)

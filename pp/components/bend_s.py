import numpy as np

import pp
from pp.components.bezier import bezier


__version__ = "0.0.2"


@pp.autoname
def bend_s(width=0.5, height=2, length=10, layer=pp.LAYER.WG, nb_points=99):
    """ S bend
    Based on bezier curve

    Args:
        width
        height: in y direction
        length: in x direction
        layer: gds number
        nb_points: number of points

    .. plot::
      :include-source:

      import pp

      c = pp.c.bend_s(width=0.5, height=20)
      pp.plotgds(c)

    """
    l, h = length, height
    c = bezier(
        width=width,
        control_points=[(0, 0), (l / 2, 0), (l / 2, h), (l, h)],
        t=np.linspace(0, 1, nb_points),
        layer=layer,
    )
    c.add_port(name="W0", port=c.ports.pop("0"))
    c.add_port(name="E0", port=c.ports.pop("1"))

    # c.ports["W0"] = c.ports.pop("0")
    # c.ports["E0"] = c.ports.pop("1")
    return c


@pp.ports.port_naming.deco_rename_ports
@pp.autoname
def bend_s_biased(width=0.5, height=2, length=10, layer=pp.LAYER.WG, nb_points=99):
    l, h = length, height
    return bezier(
        width=pp.bias.width(width),
        control_points=[(0, 0), (l / 2, 0), (l / 2, h), (l, h)],
        t=np.linspace(0, 1, nb_points),
        layer=layer,
    )


def _demo():
    c = bend_s()
    pp.write_gds(c)
    return c


if __name__ == "__main__":
    c = bend_s()
    c = bend_s_biased()
    print(c.info["min_bend_radius"])
    pp.show(c)

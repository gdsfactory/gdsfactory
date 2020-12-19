from typing import List, Tuple

import numpy as np

import pp
from pp.component import Component
from pp.components.bezier import bezier
from pp.config import conf


@pp.cell
def bend_s(
    width: float = 0.5,
    height: float = 2.0,
    length: float = 10.0,
    layer: Tuple[int, int] = pp.LAYER.WG,
    nb_points: int = 99,
    layers_cladding: List[Tuple[int, int]] = (pp.LAYER.WGCLAD),
    cladding_offset: float = conf.tech.cladding_offset,
) -> Component:
    """S bend with bezier curve

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

    y = cladding_offset
    points = [
        [c.xmin, c.ymin - y],
        [c.xmax, c.ymin - y],
        [c.xmax, c.ymax + y],
        [c.xmin, c.ymax + y],
    ]
    for layer in layers_cladding:
        c.add_polygon(points, layer=layer)

    # c.ports["W0"] = c.ports.pop("0")
    # c.ports["E0"] = c.ports.pop("1")
    return c


@pp.cell
def bend_s_biased(width=0.5, height=2, length=10, layer=pp.LAYER.WG, nb_points=99):
    l, h = length, height
    return bezier(
        width=pp.bias.width(width),
        control_points=[(0, 0), (l / 2, 0), (l / 2, h), (l, h)],
        t=np.linspace(0, 1, nb_points),
        layer=layer,
    )


if __name__ == "__main__":
    c = bend_s()
    c.pprint()
    # c = bend_s_biased()
    # print(c.info["min_bend_radius"])
    pp.show(c)

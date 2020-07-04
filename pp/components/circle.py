from typing import Tuple
import numpy as np
from numpy import cos, sin, pi

import pp
from pp.component import Component


@pp.autoname
def circle(
    radius: float = 10.0,
    angle_resolution: float = 2.5,
    layer: Tuple[int, int] = pp.LAYER.WG,
) -> Component:
    """ Generate a circle geometry.

    Args:
        radius: float, Radius of the circle.
        angle_resolution: float, Resolution of the curve of the ring (# of degrees per point).
        layer: (int, array-like[2], or set) Specific layer(s) to put polygon geometry on.

    .. plot::
      :include-source:

      import pp

      c = pp.c.circle(radius = 10, angle_resolution = 2.5, layer = 0)
      pp.plotgds(c)

    """

    c = pp.Component()
    t = np.linspace(0, 360, int(360 / angle_resolution) + 1) * pi / 180
    xpts = (radius * cos(t)).tolist()
    ypts = (radius * sin(t)).tolist()
    c.add_polygon(points=(xpts, ypts), layer=layer)
    return c


if __name__ == "__main__":
    c = circle()
    pp.show(c)

from typing import Tuple
import numpy as np
from numpy import pi, cos, sin

import pp
from pp.component import Component


@pp.autoname
def ring(
    radius: float = 10.0,
    width: float = 0.5,
    angle_resolution: float = 2.5,
    layer: Tuple[int, int] = pp.LAYER.WG,
) -> Component:
    """ Returns ring geometry.
    The ring is formed by taking the radius out to the specified value, and then constructing the thickness by dividing the width in half and adding that value to either side of the radius.
    The angle_resolution alters the precision of the curve of the ring.
    Larger values yield lower resolution

    Args:
        radius: (float) Middle radius of the ring
        width: (float) Width of the ring
        angle_resolution: (float) Resolution of the curve of the ring (# of degrees per point)
        layer: (int, array-like[2], or set) Specific layer(s) to put polygon geometry on

    .. plot::
      :include-source:

      import pp

      c = pp.c.ring(radius=10, width=0.5, angle_resolution=2.5, layer=0)
      pp.plotgds(c)

    """

    D = pp.Component()
    inner_radius = radius - width / 2
    outer_radius = radius + width / 2
    n = int(np.round(360 / angle_resolution))
    t = np.linspace(0, 360, n + 1) * pi / 180
    inner_points_x = (inner_radius * cos(t)).tolist()
    inner_points_y = (inner_radius * sin(t)).tolist()
    outer_points_x = (outer_radius * cos(t)).tolist()
    outer_points_y = (outer_radius * sin(t)).tolist()
    xpts = inner_points_x + outer_points_x[::-1]
    ypts = inner_points_y + outer_points_y[::-1]
    D.add_polygon(points=(xpts, ypts), layer=layer)
    return D


if __name__ == "__main__":
    c = ring()
    pp.show(c)

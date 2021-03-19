from typing import Tuple

import numpy as np
from numpy import cos, pi, sin

import pp
from pp.component import Component


@pp.cell
def ring(
    radius: float = 10.0,
    width: float = 0.5,
    angle_resolution: float = 2.5,
    layer: Tuple[int, int] = pp.LAYER.WG,
) -> Component:
    """Returns a ring.

    Args:
        radius: ring radius
        width: of the ring
        angle_resolution: number of points per degree
        layer: layer

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
    c.show()

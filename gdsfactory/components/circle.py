from typing import Tuple

import numpy as np
from numpy import cos, pi, sin

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def circle(
    radius: float = 10.0,
    angle_resolution: float = 2.5,
    layer: Tuple[int, int] = gf.LAYER.WG,
) -> Component:
    """Generate a circle geometry.

    Args:
        radius: of the circle.
        angle_resolution: number of degrees per point.
        layer: layer.

    """

    c = Component()
    t = np.linspace(0, 360, int(360 / angle_resolution) + 1) * pi / 180
    xpts = (radius * cos(t)).tolist()
    ypts = (radius * sin(t)).tolist()
    c.add_polygon(points=(xpts, ypts), layer=layer)
    return c


if __name__ == "__main__":
    c = circle()
    c.show()

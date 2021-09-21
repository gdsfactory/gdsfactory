from typing import Tuple

import numpy as np
from numpy import cos, pi, sin, sqrt

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def ellipse(
    radii: Tuple[float, float] = (10.0, 5.0),
    angle_resolution: float = 2.5,
    layer: Tuple[int, int] = gf.LAYER.WG,
) -> Component:
    """Generate an ellipse geometry.

    Args:
        radii: Semimajor and semiminor axis lengths of the ellipse.
        angle_resolution: Resolution of the curve of the ring (# of degrees per point).
        layer: Specific layer(s) to put polygon geometry on.

    The orientation of the ellipse is determined by the order of the radii variables;
    if the first element is larger, the ellipse will be horizontal and if the second
    element is larger, the ellipse will be vertical.

    """
    D = gf.Component()
    a = radii[0]
    b = radii[1]
    t = np.linspace(0, 360, int(360 / angle_resolution) + 1) * pi / 180
    r = a * b / (sqrt((b * cos(t)) ** 2 + (a * sin(t)) ** 2))
    xpts = r * cos(t)
    ypts = r * sin(t)
    D.add_polygon(points=(xpts, ypts), layer=layer)
    return D


if __name__ == "__main__":
    c = ellipse()
    c.show()

from typing import Tuple

import numpy as np
from numpy import cos, pi, sin, sqrt

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.types import LayerSpec


@gf.cell
def ellipse(
    radii: Tuple[float, float] = (10.0, 5.0),
    angle_resolution: float = 2.5,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns ellipse component.

    Args:
        radii: Semimajor and semiminor axis lengths of the ellipse.
        angle_resolution: number of degrees per point.
        layer: Specific layer(s) to put polygon geometry on.

    The orientation of the ellipse is determined by the order of the radii variables;
    if the first element is larger, the ellipse will be horizontal and if the second
    element is larger, the ellipse will be vertical.
    """
    c = gf.Component()
    a = radii[0]
    b = radii[1]
    t = np.linspace(0, 360, int(360 / angle_resolution) + 1) * pi / 180
    r = a * b / (sqrt((b * cos(t)) ** 2 + (a * sin(t)) ** 2))
    xpts = r * cos(t)
    ypts = r * sin(t)
    c.add_polygon(points=(xpts, ypts), layer=layer)
    return c


if __name__ == "__main__":
    c = ellipse()
    c.show(show_ports=True)

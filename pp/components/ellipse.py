from typing import Tuple

import numpy as np
from numpy import cos, pi, sin, sqrt

import pp
from pp.component import Component
from pp.types import Number


@pp.cell
def ellipse(
    radii: Tuple[Number, Number] = (10.0, 5.0),
    angle_resolution: float = 2.5,
    layer: Tuple[int, int] = pp.LAYER.WG,
) -> Component:
    """Generate an ellipse geometry.

    Args:
        radii: (tuple) Semimajor and semiminor axis lengths of the ellipse.
        angle_resolution: (float) Resolution of the curve of the ring (# of degrees per point).
        layer: (int, array-like[2], or set) Specific layer(s) to put polygon geometry on.

    The orientation of the ellipse is determined by the order of the radii variables;
    if the first element is larger, the ellipse will be horizontal and if the second
    element is larger, the ellipse will be vertical.

    .. plot::
      :include-source:

      import pp

      c = pp.components.ellipse(radii=(10, 5), angle_resolution=2.5, layer=0)
      c.plot()

    """

    D = pp.Component()
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

from __future__ import annotations

import numpy as np
from numpy import cos, pi, sin

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell
def circle(
    radius: float = 10.0,
    angle_resolution: float = 2.5,
    layer: LayerSpec = "WG",
) -> Component:
    """Generate a circle geometry.

    Args:
        radius: of the circle.
        angle_resolution: number of degrees per point.
        layer: layer.
    """
    if radius <= 0:
        raise ValueError(f"radius={radius} must be > 0")
    c = Component()
    t = np.linspace(0, 360, int(360 / angle_resolution) + 1) * pi / 180
    xpts = (radius * cos(t)).tolist()
    ypts = (radius * sin(t)).tolist()
    c.add_polygon(points=list(zip(xpts, ypts)), layer=layer)
    return c


if __name__ == "__main__":
    c = circle()
    c.show()

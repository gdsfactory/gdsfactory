from __future__ import annotations

import numpy as np
from numpy import cos, pi, sin

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell
def ring(
    radius: float = 10.0,
    width: float = 0.5,
    angle_resolution: float = 2.5,
    layer: LayerSpec = "WG",
    angle: float | None = 360,
) -> Component:
    """Returns a ring.

    Args:
        radius: ring radius.
        width: of the ring.
        angle_resolution: number of points per degree.
        layer: layer.
        angle: angular coverage of the ring
    """
    D = gf.Component()
    inner_radius = radius - width / 2
    outer_radius = radius + width / 2
    n = int(np.round(360 / angle_resolution))
    t = np.linspace(0, angle, n + 1) * pi / 180
    inner_points_x = (inner_radius * cos(t)).tolist()
    inner_points_y = (inner_radius * sin(t)).tolist()
    outer_points_x = (outer_radius * cos(t)).tolist()
    outer_points_y = (outer_radius * sin(t)).tolist()
    xpts = inner_points_x + outer_points_x[::-1]
    ypts = inner_points_y + outer_points_y[::-1]
    D.add_polygon(points=(xpts, ypts), layer=layer)
    return D


if __name__ == "__main__":
    c = ring(radius=5, angle=270)
    c.show(show_ports=True)

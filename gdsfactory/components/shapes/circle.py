from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
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
    num_points = int(np.round(360.0 / angle_resolution)) + 1
    theta = np.deg2rad(np.linspace(0, 360, num_points, endpoint=True))
    points = np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=-1)
    c.add_polygon(points=points, layer=layer)
    return c


if __name__ == "__main__":
    c = circle()
    c.show()

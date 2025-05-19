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
    # Precompute constants and avoid extra assignment
    num_points = int(360.0 / angle_resolution + 0.5) + 1  # faster than np.round
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
    # Direct usage of numpy trigonometric methods, avoiding recomputation
    rcos = np.cos(theta)
    rsin = np.sin(theta)
    points = np.empty((num_points, 2), dtype=float)
    points[:, 0] = radius * rcos
    points[:, 1] = radius * rsin
    c.add_polygon(points=points, layer=layer)
    return c


if __name__ == "__main__":
    c = circle()
    c.show()

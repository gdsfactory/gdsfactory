from __future__ import annotations

__all__ = ["star"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def star(
    inner_radius: float = 5.0,
    outer_radius: float = 10.0,
    n_points: int = 5,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a star shape with alternating inner and outer radii.

    Args:
        inner_radius: radius of inner vertices.
        outer_radius: radius of outer vertices.
        n_points: number of star points.
        layer: layer spec.
    """
    if n_points < 3:
        raise ValueError(f"n_points={n_points} must be >= 3")
    if inner_radius <= 0 or outer_radius <= 0:
        raise ValueError("radii must be > 0")

    c = Component()
    angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False)
    points = []
    for i, a in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        points.append((r * np.cos(a), r * np.sin(a)))
    c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    c = star()
    c.show()

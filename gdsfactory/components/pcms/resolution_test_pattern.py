from __future__ import annotations

__all__ = ["resolution_test_pattern"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags={"type": "pcms"})
def resolution_test_pattern(
    radius: float = 50.0,
    n_spokes: int = 36,
    width: float = 1.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Radial Siemens star resolution test pattern.

    Creates a circle of alternating filled/empty pie-shaped wedges.
    The pattern is useful for evaluating lithographic resolution,
    as the feature size decreases toward the center.

    Args:
        radius: Outer radius of the star pattern in um.
        n_spokes: Total number of spokes (filled + empty). Must be even.
        width: Target outer edge width of each spoke in um (informational;
            actual angular width is 360/n_spokes degrees).
        layer: Layer specification for the filled wedges.
    """
    c = Component()

    angle_step = 2 * np.pi / n_spokes
    n_arc_pts = 64  # points along each wedge arc for smoothness

    for i in range(0, n_spokes, 2):
        theta_start = i * angle_step
        theta_end = theta_start + angle_step

        arc_angles = np.linspace(theta_start, theta_end, n_arc_pts)
        arc_x = radius * np.cos(arc_angles)
        arc_y = radius * np.sin(arc_angles)

        points = [(0.0, 0.0)]
        points.extend(zip(arc_x, arc_y, strict=False))
        points.append((0.0, 0.0))

        c.add_polygon(points, layer=layer)

    return c


if __name__ == "__main__":
    c = resolution_test_pattern()
    c.show()

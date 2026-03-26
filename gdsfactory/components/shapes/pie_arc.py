from __future__ import annotations

__all__ = ["pie_arc"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def pie_arc(
    radius: float = 10.0,
    radius_y: float | None = None,
    start_angle: float = 0.0,
    end_angle: float = 90.0,
    angle_resolution: float = 2.5,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a pie-shaped arc (sector/wedge), centered at origin.

    The shape is a closed polygon from the origin along two radial lines
    connected by an elliptical arc.

    Args:
        radius: x-radius (or uniform radius if radius_y is None).
        radius_y: y-radius for elliptical arc. Defaults to radius.
        start_angle: start angle in degrees.
        end_angle: end angle in degrees.
        angle_resolution: degrees per arc point.
        layer: layer spec.
    """
    if radius_y is None:
        radius_y = radius

    c = Component()
    sweep = end_angle - start_angle
    n_points = max(int(abs(sweep) / angle_resolution), 2)
    theta = np.deg2rad(np.linspace(start_angle, end_angle, n_points, endpoint=True))

    arc_points = list(
        zip(radius * np.cos(theta), radius_y * np.sin(theta), strict=False)
    )
    points = [(0, 0)] + arc_points
    c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    c = pie_arc()
    c.show()

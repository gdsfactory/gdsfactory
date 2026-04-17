from __future__ import annotations

__all__ = ["ring"]

import numpy as np
from numpy import cos, pi, sin

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["rings"])
def ring(
    radius: float = 10.0,
    width: float = 0.5,
    angle_resolution: float = 2.5,
    layer: LayerSpec = "WG",
    angle: float = 360,
    distance_resolution: float | None = None,
) -> Component:
    """Returns a ring.

    Args:
        radius: ring radius.
        width: of the ring.
        angle_resolution: max number of degrees per point.
        layer: layer.
        angle: angular coverage of the ring
        distance_resolution: max distance between points. This is an alternate way to describe the resolution besides setting angle_resolution. If distance_resolution and angle_resolution are both set, distance_resolution determines the resolution.
    """
    if radius < width / 2:
        raise ValueError(
            f"Error: radius is {radius} and width is {width}. radius must be >= width / 2."
        )

    if width < 0:
        raise ValueError(f"Error: width is {width}, but it must be nonnegative.")

    if angle > 360 or angle < 0:
        raise ValueError(f"Error: angle is {angle}, but it must be in [0, 360].")

    if distance_resolution is not None and distance_resolution <= 0:
        raise ValueError(
            f"Error: distance_resolution is {distance_resolution}, but it must be positive if given."
        )

    if distance_resolution is None and angle_resolution <= 0:
        raise ValueError(
            f"Error: angle_resolution is {angle_resolution}, but it must be positive."
        )

    D = gf.Component()
    inner_radius = radius - width / 2
    outer_radius = radius + width / 2
    if distance_resolution is not None:
        num_points = int(
            np.ceil(2 * pi * outer_radius * angle / 360 / distance_resolution)
        )
    else:
        num_points = int(np.ceil(angle / angle_resolution))
    t = np.linspace(0, angle, num_points + 1) * pi / 180
    inner_points_x = inner_radius * cos(t)
    inner_points_y = inner_radius * sin(t)
    outer_points_x = outer_radius * cos(t)
    outer_points_y = outer_radius * sin(t)
    xpts = np.concatenate([inner_points_x, outer_points_x[::-1]])
    ypts = np.concatenate([inner_points_y, outer_points_y[::-1]])
    D.add_polygon(points=list(zip(xpts, ypts, strict=False)), layer=layer)
    return D

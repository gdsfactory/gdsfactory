from __future__ import annotations

__all__ = ["circle_wave"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def circle_wave(
    radius: float = 10.0,
    amplitude: float = 1.0,
    n_oscillations: int = 8,
    angle_resolution: float = 1.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a circle with a sinusoidal boundary variation.

    The boundary radius varies as r(theta) = radius + amplitude * sin(n * theta).

    Args:
        radius: mean radius.
        amplitude: amplitude of sinusoidal modulation.
        n_oscillations: number of oscillations around the boundary.
        angle_resolution: degrees per point.
        layer: layer spec.
    """
    c = Component()
    n_points = int(np.round(360.0 / angle_resolution)) + 1
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=True)
    r = radius + amplitude * np.sin(n_oscillations * theta)
    points = np.stack((r * np.cos(theta), r * np.sin(theta)), axis=-1)
    c.add_polygon(points=points, layer=layer)
    return c


if __name__ == "__main__":
    c = circle_wave()
    c.show()

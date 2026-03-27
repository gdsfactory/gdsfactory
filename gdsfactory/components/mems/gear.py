from __future__ import annotations

__all__ = ["gear"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def gear(
    n_teeth: int = 20,
    module_size: float = 2.0,
    pressure_angle: float = 20.0,
    hub_radius: float | None = None,
    hub_hole_radius: float = 0.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a gear with simplified trapezoidal teeth.

    Args:
        n_teeth: number of teeth.
        module_size: gear module (pitch diameter / number of teeth).
        pressure_angle: pressure angle in degrees.
        hub_radius: radius of the central hub disc. Defaults to root_radius * 0.6.
        hub_hole_radius: radius of a center hole (0 to disable).
        layer: layer spec.
    """
    c = Component()

    pitch_radius = n_teeth * module_size / 2
    outer_radius = pitch_radius + module_size
    root_radius = pitch_radius - 1.25 * module_size

    if hub_radius is None:
        hub_radius = root_radius * 0.6

    pa_rad = np.radians(pressure_angle)

    # Angular pitch (full tooth + gap)
    angular_pitch = 2 * np.pi / n_teeth
    # Half-tooth angular width at pitch circle
    half_tooth_angle = angular_pitch / 4

    # Build gear outline as polygon points
    points = []

    for i in range(n_teeth):
        theta = i * angular_pitch

        # Tooth tip is narrower, root is wider based on pressure angle
        tip_half_angle = half_tooth_angle - np.tan(pa_rad) * module_size / pitch_radius
        root_half_angle = (
            half_tooth_angle + np.tan(pa_rad) * 1.25 * module_size / pitch_radius
        )

        # Root start (leading edge)
        a = theta - root_half_angle
        points.append((root_radius * np.cos(a), root_radius * np.sin(a)))

        # Tooth tip leading edge
        a = theta - tip_half_angle
        points.append((outer_radius * np.cos(a), outer_radius * np.sin(a)))

        # Tooth tip trailing edge
        a = theta + tip_half_angle
        points.append((outer_radius * np.cos(a), outer_radius * np.sin(a)))

        # Root end (trailing edge)
        a = theta + root_half_angle
        points.append((root_radius * np.cos(a), root_radius * np.sin(a)))

    c.add_polygon(points, layer=layer)

    # Hub disc
    n_hub_pts = 64
    hub_angles = np.linspace(0, 2 * np.pi, n_hub_pts, endpoint=False)
    hub_points = [(hub_radius * np.cos(a), hub_radius * np.sin(a)) for a in hub_angles]
    c.add_polygon(hub_points, layer=layer)

    # Center hole cutout
    if hub_hole_radius > 0:
        n_hole_pts = 64
        hole_angles = np.linspace(0, 2 * np.pi, n_hole_pts, endpoint=False)
        hole_points = [
            (hub_hole_radius * np.cos(a), hub_hole_radius * np.sin(a))
            for a in hole_angles
        ]
        c.add_polygon(hole_points, layer=layer)

    return c


if __name__ == "__main__":
    c = gear()
    c.show()

from __future__ import annotations

__all__ = ["torus"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags={"type": "shapes"})
def torus(
    inner_radius: float = 5.0,
    outer_radius: float = 10.0,
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    angle_resolution: float = 2.5,
    layer: LayerSpec = "WG",
    port_type: str | None = None,
) -> Component:
    """Returns a torus (annular sector / ring sector) centered at origin.

    Args:
        inner_radius: inner radius.
        outer_radius: outer radius.
        start_angle: start angle in degrees.
        end_angle: end angle in degrees.
        angle_resolution: degrees per arc point.
        layer: layer spec.
        port_type: None, optical, or electrical.
    """
    if inner_radius < 0:
        raise ValueError(f"inner_radius={inner_radius} must be >= 0")
    if outer_radius <= inner_radius:
        raise ValueError("outer_radius must be > inner_radius")

    c = Component()
    sweep = end_angle - start_angle
    n_points = max(int(abs(sweep) / angle_resolution), 2) + 1
    theta = np.deg2rad(np.linspace(start_angle, end_angle, n_points, endpoint=True))

    outer_x = outer_radius * np.cos(theta)
    outer_y = outer_radius * np.sin(theta)
    inner_x = inner_radius * np.cos(theta[::-1])
    inner_y = inner_radius * np.sin(theta[::-1])

    points = list(
        zip(
            np.concatenate([outer_x, inner_x]),
            np.concatenate([outer_y, inner_y]),
            strict=False,
        )
    )
    c.add_polygon(points, layer=layer)

    if port_type and abs(sweep) < 360:
        width = outer_radius - inner_radius
        mid_r = (inner_radius + outer_radius) / 2
        prefix = "o" if port_type == "optical" else "e"
        sa_rad = np.deg2rad(start_angle)
        ea_rad = np.deg2rad(end_angle)
        c.add_port(
            f"{prefix}1",
            center=(mid_r * np.cos(sa_rad), mid_r * np.sin(sa_rad)),
            width=width,
            orientation=start_angle + 90,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}2",
            center=(mid_r * np.cos(ea_rad), mid_r * np.sin(ea_rad)),
            width=width,
            orientation=end_angle - 90,
            layer=layer,
            port_type=port_type,
        )
        c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = torus()
    c.show()

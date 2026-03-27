from __future__ import annotations

__all__ = ["rounded_rectangle"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def rounded_rectangle(
    width: float = 20.0,
    height: float = 10.0,
    corner_radius_x: float = 3.0,
    corner_radius_y: float | None = None,
    n_corner_points: int = 20,
    layer: LayerSpec = "WG",
    port_type: str | None = None,
) -> Component:
    """Returns a rectangle with rounded corners, centered at origin.

    Args:
        width: total width of the rectangle.
        height: total height of the rectangle.
        corner_radius_x: x-radius of the corner arcs.
        corner_radius_y: y-radius of the corner arcs. Defaults to corner_radius_x.
        n_corner_points: number of points per corner arc.
        layer: layer spec.
        port_type: None, optical, or electrical.
    """
    if corner_radius_y is None:
        corner_radius_y = corner_radius_x

    rx = min(corner_radius_x, width / 2)
    ry = min(corner_radius_y, height / 2)

    c = Component()
    hw = width / 2
    hh = height / 2

    # Trace corners CCW. Each corner is a quarter-ellipse arc centered at the
    # corner's center point. The four arcs span continuous angular ranges:
    #   top-right:    0      → pi/2
    #   top-left:     pi/2   → pi
    #   bottom-left:  pi     → 3*pi/2
    #   bottom-right: 3*pi/2 → 2*pi
    corners = [
        (hw - rx, hh - ry, 0, np.pi / 2),  # top-right
        (-hw + rx, hh - ry, np.pi / 2, np.pi),  # top-left
        (-hw + rx, -hh + ry, np.pi, 3 * np.pi / 2),  # bottom-left
        (hw - rx, -hh + ry, 3 * np.pi / 2, 2 * np.pi),  # bottom-right
    ]

    points: list[tuple[float, float]] = []
    for cx, cy, a_start, a_end in corners:
        t = np.linspace(a_start, a_end, n_corner_points, endpoint=True)
        points.extend((cx + rx * np.cos(ti), cy + ry * np.sin(ti)) for ti in t)

    c.add_polygon(points, layer=layer)

    if port_type:
        prefix = "o" if port_type == "optical" else "e"
        c.add_port(
            f"{prefix}1",
            center=(hw, 0),
            width=height - 2 * ry,
            orientation=0,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}2",
            center=(-hw, 0),
            width=height - 2 * ry,
            orientation=180,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}3",
            center=(0, hh),
            width=width - 2 * rx,
            orientation=90,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}4",
            center=(0, -hh),
            width=width - 2 * rx,
            orientation=270,
            layer=layer,
            port_type=port_type,
        )
        c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = rounded_rectangle()
    c.show()

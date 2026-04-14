from __future__ import annotations

__all__ = ["arrow_junction"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags={"type": "microfluidics"})
def arrow_junction(
    main_width: float = 1.0,
    branch_width: float = 1.0,
    main_length: float = 20.0,
    branch_length: float = 10.0,
    branch_angle: float = 35.0,
    reservoir_radius: float = 0.0,
    n_reservoir_points: int = 64,
    layer: LayerSpec = "WG",
    port_type: str | None = "optical",
) -> Component:
    """Returns a microfluidic arrow (Y) junction.

    A main horizontal channel on the right side with two angled branches
    diverging to the left at +/- branch_angle from horizontal. The
    junction point is at the origin.

    Args:
        main_width: width of the main horizontal channel.
        branch_width: width of each angled branch channel.
        main_length: length of the main channel (extends to the right).
        branch_length: length of each angled branch.
        branch_angle: angle of each branch from horizontal (degrees).
        reservoir_radius: radius of circular reservoirs at endpoints (0 to disable).
        n_reservoir_points: number of polygon points for reservoir circles.
        layer: layer spec.
        port_type: None, optical, or electrical.
    """
    c = Component()
    mhw = main_width / 2
    bhw = branch_width / 2
    angle_rad = np.radians(branch_angle)

    # Main horizontal channel from origin going right
    c.add_polygon(
        [
            (0, -mhw),
            (main_length, -mhw),
            (main_length, mhw),
            (0, mhw),
        ],
        layer=layer,
    )

    # Direction vectors for each branch (going to the left)
    for sign in [1, -1]:
        a = sign * angle_rad
        # Branch direction (pointing left/away from junction)
        dx = -np.cos(a)
        dy = np.sin(a)
        # Perpendicular to branch direction
        nx = -dy
        ny = dx

        # Four corners of branch rectangle
        x0 = 0.0
        y0 = 0.0
        x1 = x0 + branch_length * dx
        y1 = y0 + branch_length * dy

        pts = [
            (x0 + bhw * nx, y0 + bhw * ny),
            (x0 - bhw * nx, y0 - bhw * ny),
            (x1 - bhw * nx, y1 - bhw * ny),
            (x1 + bhw * nx, y1 + bhw * ny),
        ]
        c.add_polygon(pts, layer=layer)

    # Branch endpoint positions
    upper_end_x = -branch_length * np.cos(angle_rad)
    upper_end_y = branch_length * np.sin(angle_rad)
    lower_end_x = -branch_length * np.cos(angle_rad)
    lower_end_y = -branch_length * np.sin(angle_rad)

    # Reservoir circles at endpoints
    if reservoir_radius > 0:
        angles = np.linspace(0, 2 * np.pi, n_reservoir_points, endpoint=False)
        for cx, cy in [
            (main_length, 0),
            (upper_end_x, upper_end_y),
            (lower_end_x, lower_end_y),
        ]:
            pts = [
                (cx + reservoir_radius * np.cos(a), cy + reservoir_radius * np.sin(a))
                for a in angles
            ]
            c.add_polygon(pts, layer=layer)

    if port_type:
        prefix = "o" if port_type == "optical" else "e"
        # Main end (right)
        c.add_port(
            f"{prefix}1",
            center=(main_length, 0),
            width=main_width,
            orientation=0,
            layer=layer,
            port_type=port_type,
        )
        # Upper branch end
        c.add_port(
            f"{prefix}2",
            center=(upper_end_x, upper_end_y),
            width=branch_width,
            orientation=180 + branch_angle,
            layer=layer,
            port_type=port_type,
        )
        # Lower branch end
        c.add_port(
            f"{prefix}3",
            center=(lower_end_x, lower_end_y),
            width=branch_width,
            orientation=180 - branch_angle,
            layer=layer,
            port_type=port_type,
        )
        c.auto_rename_ports()

    return c


if __name__ == "__main__":
    c = arrow_junction()
    c.show()

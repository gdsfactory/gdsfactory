from __future__ import annotations

__all__ = ["h_junction"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def h_junction(
    main_width: float = 1.0,
    branch_width: float = 1.0,
    main_length: float = 20.0,
    branch_length: float = 10.0,
    reservoir_radius: float = 0.0,
    n_reservoir_points: int = 64,
    layer: LayerSpec = "WG",
    port_type: str | None = "optical",
) -> Component:
    """Returns a microfluidic H-junction.

    A horizontal main channel with two vertical branches extending
    up and down from the center. Optionally adds circular reservoirs
    at the four endpoints.

    Args:
        main_width: width of the main horizontal channel.
        branch_width: width of the vertical branch channels.
        main_length: total length of the main channel.
        branch_length: length of each vertical branch.
        reservoir_radius: radius of circular reservoirs at endpoints (0 to disable).
        n_reservoir_points: number of polygon points for reservoir circles.
        layer: layer spec.
        port_type: None, optical, or electrical.
    """
    c = Component()
    mhw = main_width / 2
    bhw = branch_width / 2
    half_main = main_length / 2

    # Main horizontal channel centered at origin
    c.add_polygon(
        [
            (-half_main, -mhw),
            (half_main, -mhw),
            (half_main, mhw),
            (-half_main, mhw),
        ],
        layer=layer,
    )

    # Vertical branch going up from center
    c.add_polygon(
        [
            (-bhw, mhw),
            (bhw, mhw),
            (bhw, mhw + branch_length),
            (-bhw, mhw + branch_length),
        ],
        layer=layer,
    )

    # Vertical branch going down from center
    c.add_polygon(
        [
            (-bhw, -mhw),
            (bhw, -mhw),
            (bhw, -mhw - branch_length),
            (-bhw, -mhw - branch_length),
        ],
        layer=layer,
    )

    # Reservoir circles at endpoints
    if reservoir_radius > 0:
        angles = np.linspace(0, 2 * np.pi, n_reservoir_points, endpoint=False)
        for cx, cy in [
            (-half_main, 0),
            (half_main, 0),
            (0, mhw + branch_length),
            (0, -mhw - branch_length),
        ]:
            pts = [
                (cx + reservoir_radius * np.cos(a), cy + reservoir_radius * np.sin(a))
                for a in angles
            ]
            c.add_polygon(pts, layer=layer)

    if port_type:
        prefix = "o" if port_type == "optical" else "e"
        c.add_port(
            f"{prefix}1",
            center=(-half_main, 0),
            width=main_width,
            orientation=180,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}2",
            center=(half_main, 0),
            width=main_width,
            orientation=0,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}3",
            center=(0, mhw + branch_length),
            width=branch_width,
            orientation=90,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}4",
            center=(0, -mhw - branch_length),
            width=branch_width,
            orientation=270,
            layer=layer,
            port_type=port_type,
        )
        c.auto_rename_ports()

    return c


if __name__ == "__main__":
    c = h_junction()
    c.show()

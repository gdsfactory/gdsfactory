"""Route bundle with on_collision=None for tight port spacing."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    pitch = 2.0
    ys_left = [0, 10, 20]
    N = len(ys_left)
    ys_right = [(i - N / 2) * pitch for i in range(N)]
    layer = (1, 0)

    right_ports = [
        gf.Port(
            f"R_{i}",
            center=(0, ys_right[i]),
            width=0.5,
            orientation=180,
            layer=gf.get_layer(layer),
        )
        for i in range(N)
    ]
    left_ports = [
        gf.Port(
            f"L_{i}",
            center=(-50, ys_left[i]),
            width=0.5,
            orientation=0,
            layer=gf.get_layer(layer),
        )
        for i in range(N)
    ]

    left_ports.reverse()
    routes = gf.routing.route_bundle(
        c,
        right_ports,
        left_ports,
        radius=5,
        on_collision=None,
        cross_section="strip",
    )
    c.show()

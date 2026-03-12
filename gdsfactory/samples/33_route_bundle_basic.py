"""Route bundle connecting left and right ports with different y-spacings."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    ys_right = [0, 10, 20, 40, 50, 80]
    pitch = 127.0
    N = len(ys_right)
    ys_left = [(i - N / 2) * pitch for i in range(N)]
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
            center=(-200, ys_left[i]),
            width=0.5,
            orientation=0,
            layer=gf.get_layer(layer),
        )
        for i in range(N)
    ]

    left_ports.reverse()

    c = gf.Component()
    routes = gf.routing.route_bundle(
        c,
        left_ports,
        right_ports,
        start_straight_length=50,
        sort_ports=True,
        cross_section="strip",
    )
    c.add_ports(left_ports)
    c.add_ports(right_ports)
    c.show()

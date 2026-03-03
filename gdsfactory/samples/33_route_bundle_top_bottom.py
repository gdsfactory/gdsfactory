"""Route bundle connecting top and bottom ports vertically."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    xs_top = [0, 10, 20, 40, 50, 80]
    pitch = 127.0
    N = len(xs_top)
    xs_bottom = [(i - N / 2) * pitch for i in range(N)]
    layer = (1, 0)

    top_ports = [
        gf.Port(
            f"top_{i}",
            center=(xs_top[i], 0),
            width=0.5,
            orientation=270,
            layer=gf.get_layer(layer),
        )
        for i in range(N)
    ]

    bot_ports = [
        gf.Port(
            f"bot_{i}",
            center=(xs_bottom[i], -300),
            width=0.5,
            orientation=90,
            layer=gf.get_layer(layer),
        )
        for i in range(N)
    ]

    c = gf.Component()
    routes = gf.routing.route_bundle(
        c,
        top_ports,
        bot_ports,
        separation=5.0,
        end_straight_length=100,
        cross_section="strip",
    )
    c.show()

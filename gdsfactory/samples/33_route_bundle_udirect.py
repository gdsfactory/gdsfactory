"""Route bundle connecting ports facing the same direction (U-direct routing)."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK
from gdsfactory.port import Port

PDK.activate()


@gf.cell(cache={})
def test_connect_bundle_udirect(
    dy: int = 200, orientation: int = 270, layer: tuple[int, int] = (1, 0)
) -> gf.Component:
    xs1 = [-100, -90, -80, -55, -35, 24, 0] + [200, 210, 240]
    axis = "X" if orientation in [0, 180] else "Y"
    pitch = 10.0
    N = len(xs1)
    xs2 = [70 + i * pitch for i in range(N)]

    if axis == "X":
        ports1 = [
            Port(
                f"top_{i}",
                center=(0, xs1[i]),
                width=0.5,
                orientation=orientation,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports2 = [
            Port(
                f"bottom_{i}",
                center=(dy, xs2[i]),
                width=0.5,
                orientation=orientation,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
    else:
        ports1 = [
            Port(
                f"top_{i}",
                center=(xs1[i], 0),
                width=0.5,
                orientation=orientation,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports2 = [
            Port(
                f"bottom_{i}",
                center=(xs2[i], dy),
                width=0.5,
                orientation=orientation,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]

    c = gf.Component()
    gf.routing.route_bundle(
        c, ports1, ports2, radius=10.0, sort_ports=True, cross_section="strip"
    )
    return c


if __name__ == "__main__":
    c = test_connect_bundle_udirect()
    c.show()

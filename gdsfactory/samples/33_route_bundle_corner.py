"""Route bundle through corners connecting horizontal and vertical port groups."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK
from gdsfactory.port import Port

PDK.activate()


@gf.cell(cache={})
def test_connect_corner(N: int = 6, config: str = "A") -> gf.Component:
    d = 10.0
    sep = 5.0
    c = gf.Component()
    layer = (1, 0)

    if config in ["A", "B"]:
        a = 100.0
        ports_A_TR = [
            Port(
                f"A_TR_{i}",
                center=(d, a / 2 + i * sep),
                width=0.5,
                orientation=0,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_A_TL = [
            Port(
                f"A_TL_{i}",
                center=(-d, a / 2 + i * sep),
                width=0.5,
                orientation=180,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_A_BR = [
            Port(
                f"A_BR_{i}",
                center=(d, -a / 2 - i * sep),
                width=0.5,
                orientation=0,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_A_BL = [
            Port(
                f"A_BL_{i}",
                center=(-d, -a / 2 - i * sep),
                width=0.5,
                orientation=180,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_A = [ports_A_TR, ports_A_TL, ports_A_BR, ports_A_BL]

        ports_B_TR = [
            Port(
                f"B_TR_{i}",
                center=(a / 2 + i * sep, d),
                width=0.5,
                orientation=90,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_B_TL = [
            Port(
                f"B_TL_{i}",
                center=(-a / 2 - i * sep, d),
                width=0.5,
                orientation=90,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_B_BR = [
            Port(
                f"B_BR_{i}",
                center=(a / 2 + i * sep, -d),
                width=0.5,
                orientation=270,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_B_BL = [
            Port(
                f"B_BL_{i}",
                center=(-a / 2 - i * sep, -d),
                width=0.5,
                orientation=270,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

    elif config in ["C", "D"]:
        a = N * sep + 2 * d
        ports_A_TR = [
            Port(
                f"A_TR_{i}",
                center=(a, d + i * sep),
                width=0.5,
                orientation=0,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_A_TL = [
            Port(
                f"A_TL_{i}",
                center=(-a, d + i * sep),
                width=0.5,
                orientation=180,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_A_BR = [
            Port(
                f"A_BR_{i}",
                center=(a, -d - i * sep),
                width=0.5,
                orientation=0,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_A_BL = [
            Port(
                f"A_BL_{i}",
                center=(-a, -d - i * sep),
                width=0.5,
                orientation=180,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_A = [ports_A_TR, ports_A_TL, ports_A_BR, ports_A_BL]

        ports_B_TR = [
            Port(
                f"B_TR_{i}",
                center=(d + i * sep, a),
                width=0.5,
                orientation=90,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_B_TL = [
            Port(
                f"B_TL_{i}",
                center=(-d - i * sep, a),
                width=0.5,
                orientation=90,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_B_BR = [
            Port(
                f"B_BR_{i}",
                center=(d + i * sep, -a),
                width=0.5,
                orientation=270,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_B_BL = [
            Port(
                f"B_BL_{i}",
                center=(-d - i * sep, -a),
                width=0.5,
                orientation=270,
                layer=gf.get_layer(layer),
            )
            for i in range(N)
        ]
        ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

    if config in ["A", "C"]:
        for ports1, ports2 in zip(ports_A, ports_B, strict=True):
            gf.routing.route_bundle(
                c, ports1, ports2, radius=5, sort_ports=True, cross_section="strip"
            )
    elif config in ["B", "D"]:
        for ports1, ports2 in zip(ports_A, ports_B, strict=True):
            gf.routing.route_bundle(
                c, ports2, ports1, radius=5, sort_ports=True, cross_section="strip"
            )

    return c


if __name__ == "__main__":
    c = test_connect_corner(config="A")
    c.show()

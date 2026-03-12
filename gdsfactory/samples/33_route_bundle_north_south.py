"""Route bundle connecting north-facing ports to south-facing ports."""

from __future__ import annotations

import gdsfactory as gf


@gf.cell
def test_north_to_south(layer: tuple[int, int] = (1, 0)) -> gf.Component:
    dy = 200.0
    xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

    pitch = 10.0
    N = len(xs1)
    xs2 = [-20 + i * pitch for i in range(N // 2)]
    xs2 += [400 + i * pitch for i in range(N // 2)]

    a1 = 90
    a2 = a1 + 180

    ports1 = [
        gf.Port(
            f"top_{i}",
            center=(xs1[i], 0),
            width=0.5,
            orientation=a1,
            layer=gf.get_layer(layer),
        )
        for i in range(N)
    ]

    ports2 = [
        gf.Port(
            f"bot_{i}",
            center=(xs2[i], dy),
            width=0.5,
            orientation=a2,
            layer=gf.get_layer(layer),
        )
        for i in range(N)
    ]

    c = gf.Component()
    gf.routing.route_bundle(c, ports1, ports2, cross_section="strip")
    return c


if __name__ == "__main__":
    c = test_north_to_south()
    c.show()

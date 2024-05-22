from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory import Component, Port


@gf.cell
def test_connect_bundle_u_indirect(
    data_regression: DataRegressionFixture,
    dy=-200,
    orientation=180,
    layer=(1, 0),
):
    xs1 = [-100, -90, -80, -55, -35] + [200, 210, 240]
    axis = "X" if orientation in [0, 180] else "Y"
    pitch = 10.0
    N = len(xs1)
    xs2 = [50 + i * pitch for i in range(N)]

    a1 = orientation
    a2 = a1 + 180

    if axis == "X":
        ports1 = [
            Port(f"top_{i}", center=(0, xs1[i]), width=0.5, orientation=a1, layer=layer)
            for i in range(N)
        ]

        ports2 = [
            Port(
                f"bot_{i}",
                center=(dy, xs2[i]),
                width=0.5,
                orientation=a2,
                layer=layer,
            )
            for i in range(N)
        ]

    else:
        ports1 = [
            Port(f"top_{i}", center=(xs1[i], 0), width=0.5, orientation=a1, layer=layer)
            for i in range(N)
        ]

        ports2 = [
            Port(
                f"bot_{i}",
                center=(xs2[i], dy),
                width=0.5,
                orientation=a2,
                layer=layer,
            )
            for i in range(N)
        ]

    c = Component()
    routes = gf.routing.route_bundle(
        c,
        ports1,
        ports2,
        bend=gf.components.bend_euler,
        radius=5,
        enforce_port_ordering=False,
        sort_ports=True,
    )
    lengths = {i: route.length for i, route in enumerate(routes)}
    data_regression.check(lengths)
    return c


if __name__ == "__main__":
    c = test_connect_bundle_u_indirect(None, orientation=0)
    c.show()

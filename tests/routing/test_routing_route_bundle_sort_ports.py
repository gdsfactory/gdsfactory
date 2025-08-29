from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf


def test_route_bundle_sort_ports(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    lengths = {}
    c = gf.Component()
    ys_right = [0, 10, 20, 40, 50, 80]
    pitch = 127.0
    N = len(ys_right)
    ys_left = [(i - N / 2) * pitch for i in range(N)]
    layer = (1, 0)

    right_ports = [
        gf.Port(
            name=f"R_{i}",
            center=(0, ys_right[i]),
            width=0.5,
            orientation=180,
            layer=gf.kcl.layout.layer(*layer),
        )
        for i in range(N)
    ]
    left_ports = [
        gf.Port(
            name=f"L_{i}",
            center=(-400, ys_left[i]),
            width=0.5,
            orientation=0,
            layer=gf.kcl.layout.layer(*layer),
        )
        for i in range(N)
    ]
    left_ports.reverse()
    routes = gf.routing.route_bundle(
        c, right_ports, left_ports, sort_ports=True, cross_section="strip"
    )

    for i, route in enumerate(routes):
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)

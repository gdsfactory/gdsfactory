# type: ignore
from __future__ import annotations

from functools import partial

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory import Component, Port


def test_route_bundle_udirect_pads(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    c = gf.Component()

    pad = partial(gf.components.pad, size=(10, 10))
    pad_south = gf.components.pad_array(
        port_orientation=270, spacing=(15.0, 0), pad=pad
    )
    pt = c << pad_south
    pb = c << pad_south
    pb.drotate(90)
    pt.drotate(90)
    pb.dmove((0, -100))

    pbports = list(pb.ports)
    ptports = pt.ports

    pbports.reverse()

    routes = gf.routing.route_bundle_electrical(c, pbports, ptports, radius=5)

    lengths = {}
    for i, route in enumerate(routes):
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)


@gf.cell
def test_route_bundle_udirect(
    data_regression: DataRegressionFixture,
    check: bool = True,
    dy=200,
    orientation=270,
    layer=(1, 0),
):
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
                layer=layer,
            )
            for i in range(N)
        ]

        ports2 = [
            Port(
                f"bottom_{i}",
                center=(dy, xs2[i]),
                width=0.5,
                orientation=orientation,
                layer=layer,
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
                layer=layer,
            )
            for i in range(N)
        ]

        ports2 = [
            Port(
                f"bottom_{i}",
                center=(xs2[i], dy),
                width=0.5,
                orientation=orientation,
                layer=layer,
            )
            for i in range(N)
        ]

    c = Component()
    routes = gf.routing.route_bundle(
        c,
        ports1,
        ports2,
        radius=10.0,
        sort_ports=True,
        separation=10,
    )
    lengths = {i: route.length for i, route in enumerate(routes)}
    if check:
        data_regression.check(lengths)


if __name__ == "__main__":
    test_route_bundle_udirect_pads(None, check=False)

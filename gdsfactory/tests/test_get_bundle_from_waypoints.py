import numpy as np
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.routing.get_bundle_from_waypoints import get_bundle_from_waypoints


def test_get_bundle_from_waypointsB(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    ys1 = np.array([0, 5, 10, 15, 30, 40, 50, 60]) + 0.0
    ys2 = np.array([0, 10, 20, 30, 70, 90, 110, 120]) + 500.0
    N = ys1.size

    ports1 = [Port(f"A_{i}", (0, ys1[i]), 0.5, 0) for i in range(N)]
    ports2 = [Port(f"B_{i}", (500, ys2[i]), 0.5, 180) for i in range(N)]

    p0 = ports1[0].position + (0, 22.5)

    c = gf.Component("B")
    waypoints = [
        p0,
        p0 + (200, 0),
        p0 + (200, -200),
        p0 + (400, -200),
        (p0[0] + 400, ports2[0].y),
    ]

    routes = get_bundle_from_waypoints(ports1, ports2, waypoints)
    lengths = {}
    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
    return c


def test_get_bundle_from_waypointsC(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    ys1 = np.array([0, 5, 10, 15, 20, 60, 70, 80, 120, 125])
    ys2 = np.array([0, 5, 10, 20, 25, 30, 40, 55, 60, 65]) - 500.0
    N = ys1.size

    ports1 = [Port(f"A_{i}", (0, ys1[i]), 0.5, 0) for i in range(N)]
    ports2 = [Port(f"B_{i}", (600, ys2[i]), 0.5, 180) for i in range(N)]

    c = gf.Component()
    waypoints = [
        ports1[0].position + (200, 0),
        ports1[0].position + (200, -200),
        ports1[0].position + (400, -200),
        (ports1[0].x + 400, ports2[0].y),
        ports2[0].position,
    ]

    routes = get_bundle_from_waypoints(ports1, ports2, waypoints)
    lengths = {}
    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)

    return c


if __name__ == "__main__":
    # c = test_get_bundle_from_waypointsC(None, check=False)
    c = test_get_bundle_from_waypointsB(None, check=False)
    c.show()

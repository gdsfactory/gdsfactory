import numpy as np
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.routing.get_bundle_from_waypoints import get_bundle_from_waypoints


def test_get_bundle_from_waypointsB(
    data_regression: DataRegressionFixture,
    check: bool = True,
) -> Component:

    ys1 = np.array([0, 5, 10, 15, 30, 40, 50, 60]) + 0.0
    ys2 = np.array([0, 10, 20, 30, 70, 90, 110, 120]) + 500.0
    N = ys1.size
    layer = (1, 0)

    ports1 = [
        Port(name=f"A_{i}", center=(0, ys1[i]), width=0.5, orientation=0, layer=layer)
        for i in range(N)
    ]
    ports2 = [
        Port(
            name=f"B_{i}",
            center=(500, ys2[i]),
            width=0.5,
            orientation=180,
            layer=layer,
        )
        for i in range(N)
    ]

    p0 = ports1[0].center

    c = gf.Component("B")
    c.add_ports(ports1)
    c.add_ports(ports2)
    waypoints = [
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
    data_regression: DataRegressionFixture,
    check: bool = True,
) -> Component:

    ys1 = np.array([0, 5, 10, 15, 20, 60, 70, 80, 120, 125])
    ys2 = np.array([0, 5, 10, 20, 25, 30, 40, 55, 60, 65]) - 500.0
    N = ys1.size
    layer = (1, 0)

    ports1 = [
        Port(name=f"A_{i}", center=(0, ys1[i]), width=0.5, orientation=0, layer=layer)
        for i in range(N)
    ]
    ports2 = [
        Port(
            name=f"B_{i}",
            center=(600, ys2[i]),
            width=0.5,
            orientation=180,
            layer=layer,
        )
        for i in range(N)
    ]

    c = gf.Component()
    c.add_ports(ports1)
    c.add_ports(ports2)
    waypoints = [
        ports1[0].center + (200, 0),
        ports1[0].center + (200, -200),
        ports1[0].center + (400, -200),
        (ports1[0].x + 400, ports2[0].y),
        ports2[0].center,
    ]

    routes = get_bundle_from_waypoints(ports1, ports2, waypoints)
    lengths = {}
    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)

    return c


def test_get_bundle_from_waypoints_staggered(
    data_regression: DataRegressionFixture,
    check: bool = True,
):

    c = gf.Component()
    r = c << gf.components.array(
        component=gf.components.straight, rows=2, columns=1, spacing=(0, 20)
    )
    r.movex(60)
    r.movey(40)

    lt = c << gf.components.straight(length=15)
    lb = c << gf.components.straight(length=5)
    lt.movey(5)

    ports1 = lt.get_ports_list(orientation=0) + lb.get_ports_list(orientation=0)
    ports2 = r.get_ports_list(orientation=180)

    dx = 20
    p0 = ports1[0].center + (dx, 0)
    p1 = (ports1[0].center[0] + dx, ports2[0].center[1])
    waypoints = (p0, p1)

    routes = gf.routing.get_bundle_from_waypoints(ports1, ports2, waypoints=waypoints)
    lengths = {}
    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)

    return c


if __name__ == "__main__":
    # c = test_get_bundle_from_waypointsC(None, check=False)
    # c = test_get_bundle_from_waypointsB(None, check=False)
    c = test_get_bundle_from_waypoints_staggered(None, check=False)
    c.show(show_ports=True)

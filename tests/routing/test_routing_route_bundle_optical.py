from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf


def test_route_bundle_optical(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    lengths = {}

    c = gf.Component()

    w = c << gf.components.straight_array(n=4, spacing=200)
    d = c << gf.components.nxn(west=4, east=0)
    d.dy = w.dy
    d.dxmin = w.dxmax + 200

    ports1 = [
        w.ports["o7"],
        w.ports["o8"],
    ]
    ports2 = [
        d.ports["o2"],
        d.ports["o1"],
    ]

    routes = gf.routing.route_bundle(
        c, ports1, ports2, sort_ports=True, radius=10, cross_section="strip"
    )
    for i, route in enumerate(routes):
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)


def test_route_bundle_optical2(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    lengths = {}

    c = gf.Component()
    w = c << gf.components.straight_array(n=4, spacing=200)
    d = c << gf.components.nxn(west=4, east=1)
    d.dy = w.dy
    d.dxmin = w.dxmax + 200

    ports1 = w.ports.filter(orientation=0)
    ports2 = d.ports.filter(orientation=180)
    ports2.reverse()

    routes = gf.routing.route_bundle(
        c, ports1, ports2, sort_ports=True, cross_section="strip"
    )

    for i, route in enumerate(routes):
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)

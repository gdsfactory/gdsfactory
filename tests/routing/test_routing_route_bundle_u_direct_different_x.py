from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf


def test_route_bundle_u_direct_different_x(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    """Tests routing.route_bundle with a U shape, direct connection and different x.

    Args:
        data_regression: fixture to check the output.
        check: flag to enable regression testing.
    """
    c = gf.Component()
    w = c << gf.components.straight_array(n=4, spacing=200)
    d = c << gf.components.nxn(west=4, east=0, north=0, south=0)
    d.dy = w.dy
    d.dxmin = w.dxmax + 200

    ports1 = w.ports.filter(orientation=0)
    ports2 = d.ports.filter(orientation=0)

    ports1 = [
        w.ports["o7"],
        w.ports["o8"],
    ]
    ports2 = [
        d.ports["o2"],
        d.ports["o1"],
    ]

    routes = gf.routing.route_bundle(c, ports1, ports2, cross_section="strip")
    lengths = {}
    for i, route in enumerate(routes):
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)


if __name__ == "__main__":
    # test_route_bundle_u_direct_different_x(None, check=False)
    c = gf.Component("test_route_bundle_u_direct_different_x")
    w = c << gf.components.straight_array(n=4, spacing=200)
    d = c << gf.components.nxn(west=4, east=0, north=0, south=0)
    d.dy = w.dy
    d.dxmin = w.dxmax + 200

    ports1 = w.ports.filter(orientation=0)
    ports2 = d.ports.filter(orientation=0)

    ports1 = [
        w.ports["o7"],
        w.ports["o8"],
    ]
    ports2 = [
        d.ports["o2"],
        d.ports["o1"],
    ]

    routes = gf.routing.route_bundle(c, ports1, ports2, cross_section="strip")
    lengths = {}
    for i, route in enumerate(routes):
        lengths[i] = route.length

    c.show()

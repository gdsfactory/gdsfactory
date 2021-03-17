from pytest_regressions.data_regression import DataRegressionFixture

import pp
from pp.component import Component


def test_get_bundle_west_to_north(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    lengths = {}

    c = pp.Component("test_get_bundle_west_to_north")
    w = h = 10
    c = pp.Component()
    pad_south = pp.components.pad_array(
        port_list=["S"], spacing=(15, 0), width=w, height=h, n=3
    )
    pad_north = pp.components.pad_array(
        port_list=["N"], spacing=(15, 0), width=w, height=h, n=3
    )
    pl = c << pad_south
    pb = c << pad_north
    pl.rotate(90)
    pb.move((100, -100))

    pbports = pb.get_ports_list()
    ptports = pl.get_ports_list()

    routes = pp.routing.get_bundle(
        pbports,
        ptports,
        route_filter=pp.routing.get_route_from_waypoints_electrical,
        bend_factory=pp.components.corner,
    )
    for i, route in enumerate(routes):
        c.add(route["references"])
        lengths[i] = route["length"]

    if check:
        data_regression.check(lengths)
    return c


def test_get_bundle_west_to_north2(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    lengths = {}

    c = pp.Component("test_get_bundle_west_to_north2")
    pbottom_facing_north = pp.port_array(midpoint=(0, 0), orientation=90, delta=(30, 0))
    ptop_facing_west = pp.port_array(
        midpoint=(100, 100), orientation=180, delta=(0, -30)
    )

    routes = pp.routing.get_bundle(
        pbottom_facing_north,
        ptop_facing_west,
        route_filter=pp.routing.get_route_from_waypoints_electrical,
        bend_factory=pp.components.corner,
    )

    for i, route in enumerate(routes):
        c.add(route["references"])
        lengths[i] = route["length"]

    if check:
        data_regression.check(lengths)
    return c


if __name__ == "__main__":
    c = test_get_bundle_west_to_north(None, check=False)
    c.show()

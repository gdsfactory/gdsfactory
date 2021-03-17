from pytest_regressions.data_regression import DataRegressionFixture

import pp
from pp.component import Component
from pp.components.electrical import corner


def test_get_bundle_electrical(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    lengths = {}

    c = pp.Component("test_get_bundle")
    c1 = c << pp.components.pad()
    c2 = c << pp.components.pad()
    c2.move((200, 100))
    routes = pp.routing.get_bundle(
        [c1.ports["E"]],
        [c2.ports["W"]],
        route_filter=pp.routing.get_route_from_waypoints_electrical,
        bend_factory=corner,
    )

    for i, route in enumerate(routes):
        c.add(route["references"])
        lengths[i] = route["length"]

    routes = pp.routing.get_bundle(
        [c1.ports["S"]],
        [c2.ports["E"]],
        route_filter=pp.routing.get_route_from_waypoints_electrical,
        start_straight=20.0,
        bend_factory=corner,
    )
    for i, route in enumerate(routes):
        c.add(route["references"])
        lengths[i] = route["length"]

    if check:
        data_regression.check(lengths)
    return c


if __name__ == "__main__":
    c = test_get_bundle_electrical(None, check=False)
    c.show()

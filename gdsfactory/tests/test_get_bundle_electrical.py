from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component


def test_get_bundle_electrical(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    lengths = {}

    c = gf.Component("test_get_bundle")
    c1 = c << gf.components.pad()
    c2 = c << gf.components.pad()
    c2.move((200, 100))
    routes = gf.routing.get_bundle(
        [c1.ports["E"]],
        [c2.ports["W"]],
        bend_factory=gf.components.wire_corner,
        waveguide="metal_routing",
    )

    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    routes = gf.routing.get_bundle(
        [c1.ports["S"]],
        [c2.ports["E"]],
        start_straight=20.0,
        bend_factory=gf.components.wire_corner,
        waveguide="metal_routing",
    )
    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
    return c


if __name__ == "__main__":
    c = test_get_bundle_electrical(None, check=False)
    c.show()

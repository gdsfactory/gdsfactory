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
        [c1.ports["e3"]],
        [c2.ports["e1"]],
        bend=gf.components.wire_corner,
        width=10,
        # auto_widen=False,
        auto_widen=True,
    )
    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    routes = gf.routing.get_bundle(
        [c1.ports["e4"]],
        [c2.ports["e3"]],
        start_straight_length=20.0,
        bend=gf.components.wire_corner,
        width=10,
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

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.difftest import difftest


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


def test_get_bundle_electrical2(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    lengths = {}

    c = gf.Component("pads_bundle_steps")
    pt = c << gf.components.pad_array(
        gf.partial(gf.components.pad, size=(30, 30)),
        orientation=270,
        columns=3,
        spacing=(50, 0),
    )
    pb = c << gf.components.pad_array(orientation=0, columns=1, rows=3)
    pt.move((500, 500))

    routes = gf.routing.get_bundle_from_steps_electrical(
        pt.ports,
        pb.ports,
        end_straight_length=60,
        separation=30,
        steps=[{"dy": -50}, {"dx": -100}],
    )

    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
        difftest(c)
    return c


if __name__ == "__main__":
    # c = test_get_bundle_electrical(None, check=False)
    c = test_get_bundle_electrical2(None, check=False)
    c.show(show_ports=True)

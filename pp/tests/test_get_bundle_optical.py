from pytest_regressions.data_regression import DataRegressionFixture

import pp
from pp.component import Component


def test_get_bundle_optical(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    lengths = {}

    c = pp.Component("test_get_bundle_optical")

    w = c << pp.components.straight_array(n_straights=4, spacing=200)
    d = c << pp.components.nxn(west=4, east=1)
    d.y = w.y
    d.xmin = w.xmax + 200

    ports1 = [
        w.ports["E1"],
        w.ports["E0"],
    ]
    ports2 = [
        d.ports["W1"],
        d.ports["W0"],
    ]

    routes = pp.routing.link_optical_ports(
        ports1, ports2, sort_ports=True, bend_radius=10
    )
    for i, route in enumerate(routes):
        c.add(route["references"])
        lengths[i] = route["length"]

    if check:
        data_regression.check(lengths)
    return c


def test_get_bundle_optical2(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    lengths = {}

    c = pp.Component("test_get_bundle_optical2")
    w = c << pp.components.straight_array(n_straights=4, spacing=200)
    d = c << pp.components.nxn(west=4, east=1)
    d.y = w.y
    d.xmin = w.xmax + 200

    ports1 = w.get_ports_list(prefix="E")
    ports2 = d.get_ports_list(prefix="W")

    routes = pp.routing.link_optical_ports(ports1, ports2, sort_ports=True)

    for i, route in enumerate(routes):
        c.add(route["references"])
        lengths[i] = route["length"]

    if check:
        data_regression.check(lengths)
    return c


if __name__ == "__main__":
    # c = test_get_bundle_optical(None, check=False)
    c = test_get_bundle_optical2(None, check=False)
    c.show()

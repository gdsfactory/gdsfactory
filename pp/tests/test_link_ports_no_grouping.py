from pytest_regressions.data_regression import DataRegressionFixture

import pp
from pp.component import Component


def test_link_optical_ports_no_grouping(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    c = pp.Component("test_link_optical_ports_no_grouping")
    w = c << pp.components.straight_array(n_straights=4, spacing=200)
    d = c << pp.components.nxn()
    d.y = w.y
    d.xmin = w.xmax + 200

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

    routes = pp.routing.link_optical_ports_no_grouping(ports1, ports2, sort_ports=True)

    lengths = {}
    for i, route in enumerate(routes):
        c.add(route["references"])
        lengths[i] = route["length"]

    if check:
        data_regression.check(lengths)
    return c


if __name__ == "__main__":
    c = test_link_optical_ports_no_grouping(None, check=False)
    c.show()

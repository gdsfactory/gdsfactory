from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.routing.get_bundle import get_bundle_same_axis_no_grouping


def test_link_optical_ports_no_grouping(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    c = gf.Component("test_link_optical_ports_no_grouping")
    w = c << gf.components.straight_array(n=4, spacing=200)
    d = c << gf.components.nxn(west=4, east=1)
    d.y = w.y
    d.xmin = w.xmax + 200

    ports1 = [
        w.ports["o7"],
        w.ports["o8"],
    ]
    ports2 = [
        d.ports["o2"],
        d.ports["o1"],
    ]

    routes = get_bundle_same_axis_no_grouping(ports1, ports2, sort_ports=True)
    # routes = gf.routing.get_bundle(ports1, ports2, sort_ports=True)

    lengths = {}
    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
    return c


if __name__ == "__main__":
    c = test_link_optical_ports_no_grouping(None, check=False)
    c.show(show_ports=True)

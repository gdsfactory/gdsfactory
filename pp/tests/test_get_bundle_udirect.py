from pytest_regressions.data_regression import DataRegressionFixture

import pp
from pp.component import Component


def test_get_bundle_udirect(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:

    c = pp.Component("test_get_bundle_udirect")
    w = h = 10
    pad_south = pp.components.pad_array(
        port_list=["S"], spacing=(15, 0), width=w, height=h
    )
    pt = c << pad_south
    pb = c << pad_south
    pb.rotate(90)
    pt.rotate(90)
    pb.move((0, -100))

    pbports = pb.get_ports_list()
    ptports = pt.get_ports_list()

    pbports.reverse()

    routes = pp.routing.get_bundle(pbports, ptports, bend_radius=5)

    lengths = {}
    for i, route in enumerate(routes):
        c.add(route["references"])
        lengths[i] = route["length"]

    if check:
        data_regression.check(lengths)
    return c


if __name__ == "__main__":
    c = test_get_bundle_udirect(None, check=False)
    c.show()

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component


def test_get_bundle_u_direct_different_x(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:
    """

    .. code::

       4----5
                    ________
       3----6     4|        |
                  3|  nxn   |
       2----7     2|        |
                  1|________|
       1----8
    """

    c = gf.Component("test_get_bundle_u_direct_different_x")
    w = c << gf.components.straight_array(n=4, spacing=200)
    d = c << gf.components.nxn(west=4, east=0, north=0, south=0)
    d.y = w.y
    d.xmin = w.xmax + 200

    ports1 = w.get_ports_list(orientation=0)
    ports2 = d.get_ports_list(orientation=0)

    ports1 = [
        w.ports["o7"],
        w.ports["o8"],
    ]
    ports2 = [
        d.ports["o2"],
        d.ports["o1"],
    ]

    routes = gf.routing.get_bundle(ports1, ports2)

    lengths = {}
    for i, route in enumerate(routes):
        c.add(route.references)
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)
    return c


if __name__ == "__main__":
    c = test_get_bundle_u_direct_different_x(None, check=False)
    c.show()

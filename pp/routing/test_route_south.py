import numpy as np

import pp
from pp.component import Component


@pp.cell
def test_route_south() -> Component:
    c = pp.Component()
    cr = c << pp.c.mmi2x2()
    routes, ports = pp.routing.route_south(cr)

    lengths = [
        15.708,
        1.0,
        0.5,
        15.708,
        5.0,
        1.6499999999999968,
        15.708,
        1.0,
        0.5,
        15.708,
        5.0,
        1.6499999999999968,
    ]
    for route, length in zip(routes, lengths):
        c.add(route)
        route_length = route.parent.length
        assert np.isclose(route_length, length)
    return c


if __name__ == "__main__":
    c = test_route_south()
    pp.show(c)

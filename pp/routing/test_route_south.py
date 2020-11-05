import pp
import numpy as np


def test_route_south():
    c = pp.c.mmi2x2()
    routes, ports = pp.routing.route_south(c)
    lengths = [
        17.207963267948966,
        22.357963267948964,
        17.20796326794897,
        22.35796326794897,
    ]

    for r, length in zip(routes, lengths):
        # print(r.parent.length)
        assert np.isclose(r.parent.length, length)
    c.add(routes)
    return c


if __name__ == "__main__":
    c = test_route_south()
    pp.show(c)

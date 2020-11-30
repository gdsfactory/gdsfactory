import pp
import numpy as np


def test_route_south():
    c = pp.c.mmi2x2()
    routes, ports = pp.routing.route_south(c)
    lengths = [
        17.257963267948966,
        22.407963267948965,
        17.257963267948966,
        22.40796326794897,
    ]

    for r, length in zip(routes, lengths):
        # print(r.parent.length)
        assert np.isclose(r.parent.length, length)
    c.add(routes)
    return c


if __name__ == "__main__":
    c = test_route_south()
    pp.show(c)

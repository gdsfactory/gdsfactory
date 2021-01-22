import numpy as np

import pp


@pp.cell
def test_route_south():
    c = pp.Component()
    cr = c << pp.c.mmi2x2()
    routes, ports = pp.routing.route_south(cr)

    l1 = 17.208
    l2 = 22.358
    lengths = [l1, l2, l1, l2]
    for route, length in zip(routes, lengths):
        print(route["settings"]["length"])
        c.add(route["references"])
        assert np.isclose(route["settings"]["length"], length)
    return c


if __name__ == "__main__":
    c = test_route_south()
    pp.show(c)

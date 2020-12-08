import numpy as np

import pp


@pp.cell
def test_route_south():
    c = pp.Component()
    cr = c << pp.c.mmi2x2()
    routes, ports = pp.routing.route_south(cr)

    l1 = 17.207963267948966
    l2 = 22.35796326794896
    lengths = [l1, l2, l1, l2]
    for r, length in zip(routes, lengths):
        print(r.parent.length)

    for r, length in zip(routes, lengths):
        assert np.isclose(r.parent.length, length)
    c.add(routes)
    return c


if __name__ == "__main__":
    c = test_route_south()
    pp.show(c)

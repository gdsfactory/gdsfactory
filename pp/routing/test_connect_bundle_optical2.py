import numpy as np

import pp


def test_connect_bundle_optical2():
    """FIXME. Actual length of the route = 499
    for some reason the route length is 10um shorter than the layout.

    b = 15.708
    route_length = 10+35+95.05+35+b+35+208+35+b+15
    print(route_length) = 499.46

    route_length = 10+t+89.55+t+b+t+9.44+t+b+20.5
    print(route_length) = 300.906
    """
    c = pp.Component()

    w = c << pp.c.waveguide_array(n_waveguides=4, spacing=200)
    d = c << pp.c.nxn(west=4, east=1)
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

    r = pp.routing.link_optical_ports(ports1, ports2, sort_ports=True, bend_radius=10)

    print(r[0].parent.length)
    assert np.isclose(r[0].parent.length, 489.46592653589795, atol=0.1)

    print(r[1].parent.length)
    assert np.isclose(r[1].parent.length, 290.798926535898, atol=0.1)

    c.add(r)
    return c


if __name__ == "__main__":
    c = test_connect_bundle_optical2()
    pp.show(c)

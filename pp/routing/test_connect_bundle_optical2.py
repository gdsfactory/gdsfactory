import numpy as np

import pp
from pp.component import Component


@pp.cell
def test_connect_bundle_optical2() -> Component:
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

    routes = pp.routing.link_optical_ports(
        ports1, ports2, sort_ports=True, bend_radius=10
    )

    lengths = [489.46592653589795, 290.798926535898]

    for route, length in zip(routes, lengths):
        c.add(route["references"])
        assert np.isclose(route["length"], length, atol=0.1)

    return c


if __name__ == "__main__":
    c = test_connect_bundle_optical2()
    pp.show(c)

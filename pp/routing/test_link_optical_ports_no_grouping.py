import numpy as np

import pp
from pp.component import Component


@pp.cell
def test_link_optical_ports_no_grouping() -> Component:
    c = pp.Component()

    w = c << pp.c.waveguide_array(n_waveguides=4, spacing=200)
    d = c << pp.c.nxn()
    d.y = w.y
    d.xmin = w.xmax + 200

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

    routes = pp.routing.link_optical_ports_no_grouping(ports1, ports2, sort_ports=True)
    lengths = [489.416]
    for route, length in zip(routes, lengths):
        # print(route["length"])
        c.add(route["references"])
        assert np.isclose(route["length"], length)

    return c


if __name__ == "__main__":
    c = test_link_optical_ports_no_grouping()
    pp.show(c)

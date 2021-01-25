import numpy as np

import pp
from pp.component import Component


@pp.cell
def test_connect_bundle_u_direct_different_x() -> Component:
    """u direct with different x."""
    c = pp.Component()

    w = c << pp.c.waveguide_array(n_waveguides=4, spacing=200)
    d = c << pp.c.nxn()
    d.y = w.y
    d.xmin = w.xmax + 200

    ports1 = w.get_ports_list(prefix="E")
    ports2 = d.get_ports_list(prefix="E")

    ports1 = [
        w.ports["E0"],
        w.ports["E1"],
    ]
    ports2 = [
        d.ports["E1"],
        d.ports["E0"],
    ]

    routes = pp.routing.connect_bundle(ports1, ports2, sort_ports=True)
    lengths = [316.936, 529.269]

    for route, length in zip(routes, lengths):
        # print(route["length"])
        c.add(route["references"])
        assert np.isclose(route["length"], length)
    return c


if __name__ == "__main__":
    c = test_connect_bundle_u_direct_different_x()
    pp.show(c)

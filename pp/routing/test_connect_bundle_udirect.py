import numpy as np

import pp
from pp.component import Component


@pp.cell
def test_connect_u_direct() -> Component:
    w = h = 10
    c = pp.Component()
    pad_south = pp.c.pad_array(port_list=["S"], spacing=(15, 0), width=w, height=h)
    pt = c << pad_south
    pb = c << pad_south
    pb.rotate(90)
    pt.rotate(90)
    pb.move((0, -100))

    pbports = pb.get_ports_list()
    ptports = pt.get_ports_list()

    pbports.reverse()

    routes = pp.routing.get_bundle(pbports, ptports, bend_radius=5)
    lengths = [37.735, 77.735, 117.735, 157.735, 197.735]

    for route, length in zip(routes, lengths):
        c.add(route["references"])
        print(route["length"])
        assert np.isclose(
            route["length"], length
        ), f"{route['length']} different from {length}"

    return c


if __name__ == "__main__":
    c = test_connect_u_direct()
    c.show()

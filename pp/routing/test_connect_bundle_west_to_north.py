import numpy as np

import pp
from pp.component import Component


@pp.cell
def test_connect_bundle_west_to_north() -> Component:
    w = h = 10
    c = pp.Component()
    pad_south = pp.c.pad_array(port_list=["S"], spacing=(15, 0), width=w, height=h, n=3)
    pad_north = pp.c.pad_array(port_list=["N"], spacing=(15, 0), width=w, height=h, n=3)
    pl = c << pad_south
    pb = c << pad_north
    pl.rotate(90)
    pb.move((100, -100))

    pbports = pb.get_ports_list()
    ptports = pl.get_ports_list()

    routes = pp.routing.connect_bundle(
        pbports, ptports, route_filter=pp.routing.connect_elec_waypoints
    )

    lengths = [190, 220, 250]

    for route, length in zip(routes, lengths):
        # print(route["length"])
        c.add(route["references"])
        assert np.isclose(route["length"], length)

    return c


if __name__ == "__main__":
    c = test_connect_bundle_west_to_north()
    pp.show(c)

import numpy as np

import pp


def test_connect_bundle_west_to_north():
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
    c.add(routes)
    # print(routes[0].parent.length)
    # print(routes[1].parent.length)
    # print(routes[2].parent.length)
    assert np.isclose(routes[0].parent.length, 190.0)
    assert np.isclose(routes[1].parent.length, 220.0)
    assert np.isclose(routes[2].parent.length, 250.0)
    return c


if __name__ == "__main__":
    c = test_connect_bundle_west_to_north()
    pp.show(c)

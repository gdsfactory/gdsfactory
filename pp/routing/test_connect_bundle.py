import numpy as np

import pp
from pp.component import Component


def test_connect_bundle() -> Component:
    c = pp.Component("test_connect_bundle")
    c1 = c << pp.c.pad()
    c2 = c << pp.c.pad()
    c2.move((200, 100))
    routes = pp.routing.connect_bundle(
        [c1.ports["E"]], [c2.ports["W"]], route_filter=pp.routing.connect_elec_waypoints
    )

    route = routes[0]
    assert np.isclose(route["length"], 200)
    c.add(route["references"])

    routes = pp.routing.connect_bundle(
        [c1.ports["S"]],
        [c2.ports["E"]],
        route_filter=pp.routing.connect_elec_waypoints,
        start_straight=20.0,
    )
    route = routes[0]
    assert np.isclose(route["length"], 480.02)
    c.add(route["references"])
    return c


if __name__ == "__main__":
    c = test_connect_bundle()
    pp.show(c)

import numpy as np

import pp
from pp.component import Component


def test_get_bundle() -> Component:
    c = pp.Component("test_get_bundle")
    c1 = c << pp.c.pad()
    c2 = c << pp.c.pad()
    c2.move((200, 100))
    routes = pp.routing.get_bundle(
        [c1.ports["E"]],
        [c2.ports["W"]],
        route_filter=pp.routing.get_route_from_waypoints_electrical,
    )

    route = routes[0]
    assert np.isclose(route["length"], 200)
    c.add(route["references"])

    routes = pp.routing.get_bundle(
        [c1.ports["S"]],
        [c2.ports["E"]],
        route_filter=pp.routing.get_route_from_waypoints_electrical,
        start_straight=20.0,
    )
    route = routes[0]
    assert np.isclose(route["length"], 480.02)
    c.add(route["references"])
    return c


if __name__ == "__main__":
    c = test_get_bundle()
    c.show()

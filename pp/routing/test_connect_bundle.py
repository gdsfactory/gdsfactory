import numpy as np

import pp
from pp.component import Component
from pp.components.electrical import corner


def test_get_bundle() -> Component:
    c = pp.Component("test_get_bundle")
    c1 = c << pp.c.pad()
    c2 = c << pp.c.pad()
    c2.move((200, 100))
    routes = pp.routing.get_bundle(
        [c1.ports["E"]],
        [c2.ports["W"]],
        route_filter=pp.routing.get_route_from_waypoints_electrical,
        bend_factory=corner,
    )

    route = routes[0]
    print(route["length"])
    assert np.isclose(route["length"], 189.98)
    c.add(route["references"])

    routes = pp.routing.get_bundle(
        [c1.ports["S"]],
        [c2.ports["E"]],
        route_filter=pp.routing.get_route_from_waypoints_electrical,
        start_straight=20.0,
        bend_factory=corner,
    )
    route = routes[0]
    print(route["length"])
    assert np.isclose(route["length"], 420.0)
    c.add(route["references"])
    return c


if __name__ == "__main__":
    c = test_get_bundle()
    c.show()

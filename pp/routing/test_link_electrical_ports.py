import numpy as np

import pp
from pp.component import Component


@pp.cell
def test_link_electrical_ports() -> Component:
    """I recommend using get_bundle instead"""
    c = pp.Component("demo_get_bundle_small_electrical")
    c1 = c << pp.c.pad()
    c2 = c << pp.c.pad()
    c2.move((200, 100))
    routes = pp.routing.link_electrical_ports(
        [c1.ports["E"]],
        [c2.ports["W"]],
        route_filter=pp.routing.get_route_from_waypoints_electrical,
    )
    lengths = [209.98]
    for route, length in zip(routes, lengths):
        print(route["length"])
        c.add(route["references"])
        assert np.isclose(route["length"], length)

    routes = pp.routing.link_electrical_ports(
        [c1.ports["S"]],
        [c2.ports["E"]],
        route_filter=pp.routing.get_route_from_waypoints_electrical,
    )
    lengths = [420.0]
    for route, length in zip(routes, lengths):
        print(route["length"])
        c.add(route["references"])
        assert np.isclose(route["length"], length)
    return c


if __name__ == "__main__":
    c = test_link_electrical_ports()
    c.show()

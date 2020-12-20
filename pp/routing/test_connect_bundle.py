import numpy as np

import pp


def test_connect_bundle():
    c = pp.Component("test_connect_bundle")
    c1 = c << pp.c.pad()
    c2 = c << pp.c.pad()
    c2.move((200, 100))
    route = pp.routing.connect_bundle(
        [c1.ports["E"]], [c2.ports["W"]], route_filter=pp.routing.connect_elec_waypoints
    )
    c.add(route)
    # print(route[0].parent.length)
    assert np.isclose(route[0].parent.length, 200.0)

    route = pp.routing.connect_bundle(
        [c1.ports["S"]],
        [c2.ports["E"]],
        route_filter=pp.routing.connect_elec_waypoints,
        start_straight=20.0,
    )
    c.add(route)
    print(route[0].parent.length)
    assert np.isclose(route[0].parent.length, 480.02)
    return c


if __name__ == "__main__":
    c = test_connect_bundle()
    pp.show(c)

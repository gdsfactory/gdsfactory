import numpy as np

import pp
from pp.component import Component


def test_connect_bundle_west_to_north() -> Component:
    c = pp.Component()

    pbottom_facing_north = pp.port_array(midpoint=(0, 0), orientation=90, delta=(30, 0))
    ptop_facing_west = pp.port_array(
        midpoint=(100, 100), orientation=180, delta=(0, -30)
    )

    routes = pp.routing.connect_bundle(
        pbottom_facing_north,
        ptop_facing_west,
        route_filter=pp.routing.connect_elec_waypoints,
        # bend_radius=50
    )
    lengths = [200, 140]
    for route, length in zip(routes, lengths):
        c.add(route["references"])
        assert np.isclose(route["length"], length)

    return c


if __name__ == "__main__":
    c = test_connect_bundle_west_to_north()
    pp.show(c)

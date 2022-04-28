"""
FIXME: electrical connections should ignore port orientation
"""

import gdsfactory as gf


if __name__ == "__main__":

    c = gf.Component("ports_none_orientation")
    # c1 = c << gf.components.pad(port_orientation=0)
    # c2 = c << gf.components.pad(port_orientation=180)

    c1 = c << gf.components.pad(port_orientation=None)
    c2 = c << gf.components.pad(port_orientation=None)
    c2.movex(200)
    c2.movey(50)

    route = gf.routing.get_route_electrical(
        c1.ports["pad"],
        c2.ports["pad"],
    )
    c.add(route.references)
    c.show()

    # print(c2.ports['pad'].midpoint)

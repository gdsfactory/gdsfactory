"""
FIXME: electrical connections should ignore port orientation
"""

import gdsfactory as gf


if __name__ == "__main__":

    c = gf.Component("mzi_with_pads")
    # c1 = c << gf.components.pad(port_orientation=0)
    # c2 = c << gf.components.pad(port_orientation=180)

    c1 = c << gf.components.pad(port_orientation=None)
    c2 = c << gf.components.pad(port_orientation=None)

    c2.movex(200)

    route = gf.routing.get_route_electrical(
        c1.ports["e1"],
        c2.ports["e1"],
    )
    c.add(route.references)

    c.show()

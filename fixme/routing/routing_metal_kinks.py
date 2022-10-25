"""The current routing algorithm does not allow for small "kinks" in the wire.

FIXME.

More precisely, for the current wiring algorithms
gf.routing.get_route() and gf.routing.get_bundle()), if the wire needs to make an S-bend,
the lateral displacement of the wire must be GREATER than the wire width.
This makes tight wiring and/or very painful.
"""

import gdsfactory as gf


if __name__ == "__main__":

    c = gf.Component("mzi_with_pads")
    c1 = c << gf.components.pad()
    c2 = c << gf.components.pad()

    c2.movex(30)
    c2.movey(200)

    port1 = c1.ports["e2"]
    port2 = c2.ports["e4"]

    route = gf.routing.get_route(
        port1,
        port2,
        cross_section=gf.cross_section.metal1,
        width=50,
        bend=gf.components.wire_corner,
        with_sbend=True,
    )
    c.add(route.references)

    c.show(show_ports=True)

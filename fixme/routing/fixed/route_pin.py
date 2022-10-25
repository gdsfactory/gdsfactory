"""FIXME.

- bends in routes flip the polarity in some cases.

"""

import gdsfactory as gf


if __name__ == "__main__":
    c = gf.Component()

    length = 50
    s1 = c << gf.components.straight_pin(length=length)
    s2 = c << gf.components.straight_pin(length=length)

    s2.move((100, 50))

    route = gf.routing.get_route(
        s1.ports["o2"], s2.ports["o1"], cross_section=gf.cross_section.pin
    )
    c.add(route.references)

    c.show(show_ports=True)

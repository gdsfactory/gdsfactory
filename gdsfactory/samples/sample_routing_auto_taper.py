"""Slot sample."""

import gdsfactory as gf


@gf.cell
def sample_routing_auto_taper_slot() -> gf.Component:
    """Sample routing auto taper."""
    c = gf.Component()
    length = 1

    c1 = c << gf.c.straight(length=length, cross_section="slot")
    c2 = c << gf.c.straight(length=length, cross_section="slot", width=2)
    c2.movex(50)
    c2.movey(50)
    gf.routing.route_bundle(
        c,
        c1.ports["o2"],
        c2.ports["o1"],
        cross_section="slot",
    )
    return c


@gf.cell
def sample_routing_auto_taper_trenches() -> gf.Component:
    """Sample routing auto taper."""
    c = gf.Component()
    length = 1

    c1 = c << gf.c.straight(length=length, cross_section="rib_with_trenches")
    c2 = c << gf.c.straight(length=length, cross_section="rib_with_trenches", width=2)
    c2.movex(50)
    c2.movey(50)
    gf.routing.route_bundle(
        c,
        c1.ports["o2"],
        c2.ports["o1"],
        cross_section="rib_with_trenches",
    )
    return c


if __name__ == "__main__":
    c = sample_routing_auto_taper_trenches()
    c.show()

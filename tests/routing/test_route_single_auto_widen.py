from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.cross_section import Section
from gdsfactory.difftest import difftest

WIDTH_WIDE = 2.0

pin_m1 = partial(
    gf.cross_section.strip,
    width=0.5,
    width_wide=WIDTH_WIDE,
    sections=(
        Section(width=1, offset=2, layer=(24, 0), name="n+"),
        Section(width=1, offset=3, layer=(41, 0), name="m1"),
    ),
)

pin = partial(
    gf.cross_section.strip,
    sections=(Section(width=1, offset=2, layer=(24, 0), name="n+"),),
)


@gf.cell
def taper_pin(length: float = 5) -> gf.Component:
    trans = gf.path.transition(
        cross_section1=pin(),
        cross_section2=pin(width=WIDTH_WIDE),
        width_type="linear",
    )
    path = gf.path.straight(length=length)
    return gf.path.extrude_transition(path, trans)


def test_route_single_auto_widen() -> None:
    c = gf.Component("test_route_single_auto_widen")
    route = gf.routing.route_single_from_waypoints(
        [(0, 0), (300, 0), (300, 300), (-600, 300), (-600, -300)],
        cross_section=pin_m1,
        bend=partial(gf.components.bend_euler, cross_section=pin),
        # taper=taper_pin,
        radius=30,
    )
    c.add(route.references)
    difftest(c)


if __name__ == "__main__":
    test_route_single_auto_widen()
    # c = gf.Component()
    # route = gf.routing.route_single_from_waypoints(
    #     [(0, 0), (300, 0), (300, 300), (300, 600), (600, 600)],
    #     cross_section="strip_auto_widen",
    #     radius=30,
    # )
    # c.add(route.references)
    # c.show( )

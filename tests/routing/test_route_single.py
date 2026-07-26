from __future__ import annotations

import gdsfactory as gf


def test_route_single_propagates_cross_section_to_straights() -> None:
    component = gf.Component()
    cross_section = gf.cross_section.cross_section(width=1, layer=(2, 0))
    left = component.add_ref(gf.components.straight(cross_section=cross_section))
    right = component.add_ref(gf.components.straight(cross_section=cross_section))
    right.movex(30)

    route = gf.routing.route_single(
        component,
        left.ports["o2"],
        right.ports["o1"],
        cross_section=cross_section,
    )

    assert route.length == 20_000

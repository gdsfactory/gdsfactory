from __future__ import annotations

import gdsfactory as gf


def test_route_single_electrical_defaults() -> None:
    component = gf.Component()
    left = component << gf.components.rectangle(size=(50, 50))
    right = component << gf.components.rectangle(size=(50, 50))
    right.movex(100)

    route = gf.routing.route_single_electrical(
        component,
        left.ports["e3"],
        right.ports["e1"],
        layer="WG",
        width=10,
        auto_taper=False,
    )

    assert route.length_backbone == 50_000
    assert component.area("WG") == 5_500


def test_route_single_electrical_auto_taper() -> None:
    component = gf.Component()
    left = component << gf.components.rectangle(size=(50, 50))
    right = component << gf.components.rectangle(size=(50, 50))
    right.movex(100)

    route = gf.routing.route_single_electrical(
        component,
        left.ports["e3"],
        right.ports["e1"],
        layer="WG",
        width=50,
        auto_taper=True,
    )

    assert route.length_backbone == 50_000

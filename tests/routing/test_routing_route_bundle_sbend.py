from __future__ import annotations

import gdsfactory as gf


def test_route_bundle_sbend_electrical_north() -> None:
    c = gf.Component()
    c1 = c << gf.components.straight_heater_metal()
    p1 = c << gf.c.pad()
    p1.movey(200)
    gf.routing.route_bundle_sbend(
        c,
        [p1["e4"]],
        [c1.ports["l_e2"]],
        enforce_port_ordering=False,
        cross_section="metal3",
        port_name="e1",
        allow_width_mismatch=True,
    )

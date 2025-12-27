"""Routing bundle requires end ports to be on the same orientation but input can be any orientation."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    c = gf.Component()
    pad = gf.components.pad()

    p1 = c << pad
    p2 = c << pad
    p2.movex(500)
    router = "electrical"

    routes = gf.routing.route_bundle(
        c,
        ports1=[p1["e3"]],
        ports2=[p2["e1"]],
        sort_ports=True,
        cross_section="metal_routing",
        # router=router,
    )

    p3 = c << pad
    p4 = c << pad
    p3.movey(500)
    p3.movex(200)
    p4.movey(-500)
    p4.movex(200)

    routes = gf.routing.route_bundle(
        c,
        ports1=[p3["e3"]],
        ports2=[p4["e2"]],
        sort_ports=True,
        cross_section="metal_routing",
        # router=router,
    )

    lyrdb = c.connectivity_check(port_types=["electrical", "optical"])
    c.show(lyrdb=lyrdb)

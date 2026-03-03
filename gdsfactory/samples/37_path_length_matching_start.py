"""Sample path length matching with loops at the start.

PathLengthConfig parameters:
- element: which route (by index) to use as the reference length. All other
  routes will be extended to match. Use 0 for the first, -1 for the last.
- loop_side: which side of the route the delay loops appear on (left, right, center).
- loops: number of delay loops to insert.
- loop_position: where along the route to place the loops (start, center, end).
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    xs = gf.get_cross_section("strip")
    layer = gf.get_layer(xs.sections[0].layer)

    start_ports = [
        gf.Port(
            name=f"in{i}",
            center=(i * 200, -i * 150),
            orientation=90,
            layer=layer,
            width=0.5,
        )
        for i in range(3)
    ]
    end_ports = [
        gf.Port(
            name=f"out{i}",
            center=(x, 500),
            orientation=270,
            layer=layer,
            width=0.5,
        )
        for i, x in enumerate([230, 400, 500])
    ]

    c = gf.Component()
    routes = gf.routing.route_bundle(
        c,
        start_ports,
        end_ports,
        start_straight_length=300,
        cross_section="strip",
        path_length_matching_config={
            "element": 0,
            "loop_side": gf.routing.LoopSide.left,
            "loops": 2,
            "loop_position": gf.routing.LoopPosition.start,
        },
    )

    for sp, ep, route in zip(start_ports, end_ports, routes, strict=True):
        print(
            f"{sp.name} - {ep.name} "
            f"backbone: {route.length_backbone} length: {route.length:.3f}"
        )

    c.show()

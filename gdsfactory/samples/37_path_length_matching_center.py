"""Sample path length matching with loops at the center.

Places delay loops in the center of the route, on both sides (center loop_side),
matching to the second route (element=1).
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
            center=(i * 300, -i * 150),
            orientation=90,
            layer=layer,
            width=0.5,
        )
        for i in range(3)
    ]
    end_ports = [
        gf.Port(
            center=(x, 500),
            orientation=270,
            layer=layer,
            width=0.5,
        )
        for x in [230, 700, 1000]
    ]

    c = gf.Component()
    gf.routing.route_bundle(
        c,
        start_ports,
        end_ports,
        start_straight_length=300,
        cross_section="strip",
        path_length_matching_config={
            "element": 1,
            "loop_side": gf.routing.LoopSide.center,
            "loops": 2,
            "loop_position": gf.routing.LoopPosition.center,
        },
    )

    c.show()

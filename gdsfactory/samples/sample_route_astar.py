"""Sample demonstrating the A* router for obstacle-aware waveguide routing."""

from __future__ import annotations

import gdsfactory as gf

gf.gpdk.PDK.activate()


if __name__ == "__main__":
    c = gf.Component()
    cross_section = gf.get_cross_section("strip", radius=5)
    bend = gf.components.bend_euler

    straight = gf.components.straight(cross_section=cross_section)
    left = c << straight
    right = c << straight
    right.rotate(90)
    right.move((168, 63))

    obstacle = gf.components.rectangle(size=(250, 3), layer="M2")
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle3 = c << obstacle
    obstacle4 = c << obstacle
    obstacle4.rotate(90)
    obstacle1.ymin = 50
    obstacle1.xmin = -10
    obstacle2.xmin = 35
    obstacle3.ymin = 42
    obstacle3.xmin = 72.23
    obstacle4.xmin = 200
    obstacle4.ymin = 55

    route = gf.routing.route_astar(
        component=c,
        port1=left.ports["o1"],
        port2=right.ports["o2"],
        cross_section=cross_section,
        resolution=15,
        distance=12,
        avoid_layers=("M2",),
        bend=bend,
    )
    c.show()

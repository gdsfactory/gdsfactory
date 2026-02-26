"""Route bundle electrical with enlarged bbox to force obstacle avoidance."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    c = gf.Component()
    columns = 2
    ptop = c << gf.components.pad_array(columns=columns, port_orientation=270)
    pbot = c << gf.components.pad_array(port_orientation=270, columns=columns)
    ptop.movex(300)
    ptop.movey(300)

    obstacle = c << gf.c.rectangle(size=(300, 100), layer="M3", centered=True)
    obstacle.ymin = pbot.ymax - 10
    obstacle.xmin = pbot.xmax + 10

    routes = gf.routing.route_bundle_electrical(
        c,
        pbot.ports,
        ptop.ports,
        start_straight_length=100,
        separation=20,
        cross_section="metal_routing",
        bboxes=[
            obstacle.bbox().enlarge(10),
            pbot.bbox(),
            ptop.bbox(),
        ],
        sort_ports=True,
    )
    c.show()

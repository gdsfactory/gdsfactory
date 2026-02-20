"""Sample route_single with absolute x/y steps.

- x: sets the absolute x coordinate of the next waypoint.
- y: sets the absolute y coordinate of the next waypoint.
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    c = gf.Component()
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.move((200, 50))

    # x=100 sets the waypoint x-coordinate to 100 (absolute)
    gf.routing.route_single(
        c,
        mmi1.ports["o2"],
        mmi2.ports["o1"],
        steps=[{"x": 100}],
        cross_section="strip",
    )
    c.show()

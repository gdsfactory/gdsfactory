"""Sample route_bundle with absolute x/y steps.

- x: sets the absolute x coordinate of the next waypoint.
- y: sets the absolute y coordinate of the next waypoint.
- dx: shifts x relative to the current position.
- dy: shifts y relative to the current position.

You can combine absolute and relative in one step, e.g. {"x": 100, "dy": 20}.
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    c = gf.Component()
    c1 = c << gf.components.mmi2x2()
    c2 = c << gf.components.mmi2x2()
    c2.move((200, 240))

    # x=100 routes to absolute x=100, then dy=300 shifts 300um in y
    gf.routing.route_bundle(
        c,
        [c1.ports["o3"], c1.ports["o4"]],
        [c2.ports["o2"], c2.ports["o1"]],
        steps=[{"x": 100}, {"dy": 300}],
        separation=5.0,
        cross_section="strip",
        sort_ports=True,
    )
    c.show()

"""Sample route_bundle with non-Manhattan steps.

Steps that combine dx and dy in a single step produce a diagonal
(non-Manhattan) waypoint. The router automatically inserts corner
points to convert these into Manhattan-compatible segments.
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

    # A single step with both dx and dy creates a non-Manhattan waypoint.
    # The router splits the diagonal into horizontal + vertical segments.
    gf.routing.route_bundle(
        c,
        [c1.ports["o3"], c1.ports["o4"]],
        [c2.ports["o2"], c2.ports["o1"]],
        steps=[{"dx": 60, "dy": 100}],
        separation=5.0,
        cross_section="strip",
        sort_ports=True,
    )
    c.show()

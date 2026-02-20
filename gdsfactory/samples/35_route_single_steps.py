"""Sample route_single with steps using dx/dy (relative) and x/y (absolute).

- dx/dy: shift relative to the current position.
- x/y: set an absolute coordinate.
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

    # dx shifts the route 60um to the right from the current position
    gf.routing.route_single(
        c,
        mmi1.ports["o2"],
        mmi2.ports["o1"],
        steps=[{"dx": 60}],
        cross_section="strip",
    )
    c.show()

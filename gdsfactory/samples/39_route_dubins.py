"""Route using Dubins paths for shortest path with minimal bending."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    c = gf.Component()

    wg1 = c << gf.components.straight(length=100, width=3.2)
    wg2 = c << gf.components.straight(length=100, width=3.2)

    wg2.move((300, 50))
    wg2.rotate(45)

    route = gf.routing.route_dubins(
        c,
        port1=wg1.ports["o2"],
        port2=wg2.ports["o1"],
        cross_section=gf.cross_section.strip(width=3.2, radius=100),
    )
    c.show()

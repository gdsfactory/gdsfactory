"""Route multiple Dubins paths between two rotated multi-port components."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    c = gf.Component()

    comp1 = c << gf.components.nxn(west=0, east=10, xsize=10, ysize=100, wg_width=3.2)
    comp2 = c << gf.components.nxn(west=0, east=10, xsize=10, ysize=100, wg_width=3.2)

    comp2.rotate(30)
    comp2.move((500, -100))

    for i in range(10):
        port1_name = f"o{10 - i}"
        port2_name = f"o{i + 1}"
        gf.routing.route_dubins(
            c,
            port1=comp1.ports[port1_name],
            port2=comp2.ports[port2_name],
            cross_section=gf.cross_section.strip(width=3.2, radius=100 + i * 10),
        )
    c.show()

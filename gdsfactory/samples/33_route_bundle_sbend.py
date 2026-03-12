"""Route bundle using S-bends for tight port spacing where Manhattan routing fails."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    c1 = c << gf.components.nxn(east=3, ysize=20)
    c2 = c << gf.components.nxn(west=3)
    c2.move((80, 0))
    routes = gf.routing.route_bundle_sbend(
        c,
        c1.ports.filter(orientation=0),
        c2.ports.filter(orientation=180),
        enforce_port_ordering=False,
    )
    c.show()

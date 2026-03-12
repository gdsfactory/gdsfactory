"""Route bundle using S-bends for tight port spacing where Manhattan routing fails."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    c1 = c << gf.components.nxn(east=3, ysize=20)
    c2 = c << gf.components.nxn(east=0, west=3)
    c2.move((80, 0))
    # c2.rotate(90)

    routes = gf.routing.route_bundle(
        c,
        c1.ports.filter(orientation=0),
        c2.ports,
        sbend="bend_s",
        cross_section="strip",
    )
    c.show()

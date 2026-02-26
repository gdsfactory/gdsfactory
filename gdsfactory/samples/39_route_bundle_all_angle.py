"""Route bundle using all-angle (diagonal) routing."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    c = gf.Component()
    rows = 3
    straight = gf.c.straight
    w1 = c << gf.c.array(straight, rows=rows, columns=1, row_pitch=10)
    w2 = c << gf.c.array(straight, rows=rows, columns=1, row_pitch=10)
    w2.rotate(-30)
    w2.movex(140)
    p1 = list(w1.ports.filter(orientation=0))
    p2 = list(w2.ports.filter(orientation=150))
    p1.reverse()
    p2.reverse()

    gf.routing.route_bundle_all_angle(
        c,
        p1,
        p2,
        separation=3,
    )
    c.show()

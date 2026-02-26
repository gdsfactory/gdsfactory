"""Route bundle with a custom taper component for auto-tapering."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    c = gf.Component()
    s1 = c << gf.components.straight(width=4)
    s2 = c << gf.components.straight(width=4)
    s2.move((100, 50))
    route = gf.routing.route_bundle(
        c,
        [s1.ports["o2"]],
        [s2.ports["o1"]],
        auto_taper=True,
        cross_section="strip",
        auto_taper_taper=gf.components.taper(width1=4, width2=0.5, length=30),
    )
    c.show()

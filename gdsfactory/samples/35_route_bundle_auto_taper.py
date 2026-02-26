"""Route bundle with auto_taper to match port widths to cross-section width."""

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
    )
    c.show()

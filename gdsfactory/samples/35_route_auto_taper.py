"""Route bundle with auto_taper for width mismatch between ports."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    s1 = c << gf.components.straight()
    s2 = c << gf.components.straight(width=2)
    s2.move((40, 50))
    route = gf.routing.route_bundle(
        c,
        s1.ports["o2"],
        s2.ports["o1"],
        cross_section="strip",
        auto_taper=True,
    )
    c.show()

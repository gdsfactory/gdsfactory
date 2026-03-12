"""Route bundle connecting two mmi2x2 components."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    c1 = c << gf.components.mmi2x2()
    c2 = c << gf.components.mmi2x2()

    c2.move((100, 50))
    routes = gf.routing.route_bundle(
        c,
        [c1.ports["o4"], c1.ports["o3"]],
        [c2.ports["o1"], c2.ports["o2"]],
        radius=5,
        cross_section="strip",
    )
    c.show()

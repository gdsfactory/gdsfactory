"""Route bundle with tapers to wider widths for reduced propagation loss."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.move((100, 50))
    route = gf.routing.route_bundle(
        c,
        [mmi1.ports["o2"]],
        [mmi2.ports["o1"]],
        taper=gf.components.taper(width1=0.5, width2=2),
        cross_section="strip",
        min_straight_taper=20,
    )
    c.show()

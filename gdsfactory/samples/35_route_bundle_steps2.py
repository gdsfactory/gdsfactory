"""Sample routing with steps."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    c = gf.Component()
    c1 = c << gf.components.mmi2x2()
    c2 = c << gf.components.mmi2x2()
    c2.move((200, 240))

    gf.routing.route_bundle(
        c,
        [c1.ports["o3"], c1.ports["o4"]],
        [c2.ports["o2"], c2.ports["o1"]],
        steps=[{"dx": 60}, {"dy": 500}],
        separation=5.0,
        cross_section="strip",
        sort_ports=True,
        layer_marker=(1, 0),
    )
    c.show()

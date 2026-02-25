"""Route bundle all-angle connecting rotated mmi2x2 components."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    c = gf.Component()

    mmi = gf.components.mmi2x2(width_mmi=10, gap_mmi=3)
    mmi1 = c.add_ref_off_grid(mmi)
    mmi2 = c.add_ref_off_grid(mmi)

    mmi2.move((100, 10))
    mmi2.rotate(30)

    routes = gf.routing.route_bundle_all_angle(
        c,
        mmi1.ports.filter(orientation=0),
        [mmi2.ports["o2"], mmi2.ports["o1"]],
    )
    c.show()

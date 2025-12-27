"""Sample GS routing."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    p = gf.path.straight()

    g = gf.Section(
        width=2,
        offset=0,
        layer=(2, 0),
        port_names=("e1", "e2"),
        port_types=("electrical", "electrical"),
    )
    s0 = gf.Section(width=2, offset=-4, layer=(2, 0))
    s1 = gf.Section(width=2, offset=4, layer=(2, 0))

    x = gf.CrossSection(sections=(s0, s1, g), radius=8)
    c = gf.path.extrude(p, cross_section=x)
    pad = c

    c2 = gf.Component()
    pad1 = c2 << pad
    pad2 = c2 << pad
    pad2.move((100, 100))

    gf.routing.route_bundle(
        c2,
        [pad1.ports["e2"]],
        [pad2.ports["e1"]],
        cross_section=x,
        port_type="electrical",
        raise_on_error=True,
        # bend='bend_circular',
        # bend='wire_corner'
        bend="wire_corner45",
        # bend='wire_corner_sections'
    )
    c2.show()

"""Sample GS routing."""

from __future__ import annotations

from functools import partial

import gdsfactory as gf

gf.gpdk.PDK.activate()


if __name__ == "__main__":
    p = gf.path.straight()

    g = ((2, 0), -1.0, 1.0)
    s0 = ((2, 0), -5.0, -3.0)
    s1 = ((2, 0), 3.0, 5.0)

    # Route on the centered conductor, with ground conductors on either side.
    x = gf.cross_section.cross_section(width=None, sections=(g, s0, s1), radius=8)
    c = gf.path.extrude(p, cross_section=x, ports={0: ("e1", "e2", "electrical")})
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
        straight=partial(gf.c.straight, port_type="electrical"),
        # bend='wire_corner_sections'
    )
    c2.show()

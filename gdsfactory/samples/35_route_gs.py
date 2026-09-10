"""Sample GS routing."""

from __future__ import annotations

import gdsfactory as gf

# This example uses layer 2/0 for conductors, so bends and straights must
# create electrical ports on that layer too.
gf.Pdk(
    name="gs_example",
    base_pdks=[gf.gpdk.PDK],
    layer_port_types={(2, 0): "electrical"},
).activate()


if __name__ == "__main__":
    p = gf.path.straight()

    s0 = ((2, 0), -1.0, 1.0)
    s1 = ((2, 0), 3.0, 5.0)
    x = gf.cross_section.cross_section(width=None, sections=(s0, s1), radius=8)
    c = gf.path.extrude(p, cross_section=x, ports={0: ("g1", "g2", "electrical")})
    pad = c

    c2 = gf.Component()
    pad1 = c2 << pad
    pad2 = c2 << pad
    pad2.move((100, 100))

    gf.routing.route_bundle(
        c2,
        [pad1.ports["g2"]],
        [pad2.ports["g1"]],
        cross_section=x,
        raise_on_error=True,
        port_type="electrical",
    )
    c2.show()

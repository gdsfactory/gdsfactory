"""Small demonstration of the cross_section copy."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    gf.gpdk.PDK.activate()
    xs = gf.get_cross_section("rib_with_trenches")
    xs_wide = gf.get_cross_section(xs, width=2)
    c = gf.c.straight(cross_section=xs_wide)
    c = gf.c.extend_ports(c)
    c.show()  # show it in klayout

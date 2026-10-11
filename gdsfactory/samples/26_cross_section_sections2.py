"""Small demonstration of the cross_section."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    gf.gpdk.PDK.activate()
    c = gf.Component()

    c1 = c << gf.c.taper_strip_to_ridge()  # works
    # c1 = c << gf.c.taper_strip_to_ridge_trenches() # works
    # c1 = c << gf.c.taper_cross_section(cross_section2="rib2") # works

    xs_in = c1["o1"].cross_section
    xs_out = c1["o2"].cross_section

    ref_in = c << gf.c.straight(cross_section=xs_in)
    ref_out = c << gf.c.straight(cross_section=xs_out)

    ref_in.connect("o2", c1.ports["o1"])
    ref_out.connect("o1", c1.ports["o2"])
    c.add_port("o1", port=ref_in.ports["o1"])
    c.add_port("o2", port=ref_out.ports["o2"])
    c.show()

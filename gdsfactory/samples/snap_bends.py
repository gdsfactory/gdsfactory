"""Snap bends together."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    # gf.CONF.allow_offgrid = False
    c = gf.Component("snap_bends")
    b1 = c << gf.c.bend_euler(angle=37, add_pins=False)
    b2 = c << gf.c.bend_euler(angle=37, add_pins=False)
    b2.connect("o1", b1.ports["o2"])
    c = c.flatten_offgrid_references()
    print(b1["o2"].center)
    c.show()

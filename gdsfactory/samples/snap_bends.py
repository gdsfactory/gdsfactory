"""Snap bends together."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component("snap_bends")
    b1 = c << gf.c.bend_euler(angle=37)
    b2 = c << gf.c.bend_euler(angle=37)
    b2.connect("o1", b1.ports["o2"])
    c = c.flatten_invalid_refs()
    c.show()

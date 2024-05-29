"""Snap bends together."""

from __future__ import annotations

import gdsfactory as gf


@gf.cell(check_instances="vinstances")
def snap_bends() -> gf.Component:
    c = gf.Component("snap_bends")
    b1 = c << gf.c.bend_euler()
    b1.rotate(37)
    b2 = c << gf.c.bend_euler()
    b2.connect("o1", b1.ports["o2"])
    return c


if __name__ == "__main__":
    c = snap_bends()
    c.show()

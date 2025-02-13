"""Snap bends together."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    # gf.CONF.enforce_ports_on_grid = False
    c = gf.Component()
    b1 = c << gf.c.bend_euler(angle=37)
    b2 = c << gf.c.bend_euler(angle=37)
    b2.connect("o1", b1.ports["o2"])
    # print(b1["o2"].center)
    # c.flatten()
    # c.over_under(layer=(1, 0))
    c.show()

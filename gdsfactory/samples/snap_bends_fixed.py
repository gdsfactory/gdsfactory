"""Snap bends together."""

from __future__ import annotations

import gdsfactory as gf


@gf.vcell
def snap_bends_sample() -> gf.ComponentAllAngle:
    c = gf.ComponentAllAngle()
    b = gf.c.bend_euler_all_angle(angle=37)
    b1 = c << b
    b1.rotate(51.1)
    b2 = c << b
    b2.connect("o1", b1.ports["o2"])
    c.add_port("o1", port=b1.ports["o1"])
    c.add_port("o2", port=b2.ports["o2"])
    return c


if __name__ == "__main__":
    c = snap_bends_sample()
    c.plot()

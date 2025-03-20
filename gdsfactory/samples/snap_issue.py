"""snap issue."""

from __future__ import annotations

import gdsfactory as gf

nm = 1e-3
if __name__ == "__main__":
    c = gf.Component()
    s1 = c << gf.c.straight(length=1 + 1.5 * nm)
    s2 = c << gf.c.straight(length=1)
    s2.connect("o1", s1.ports["o2"])
    c.show()

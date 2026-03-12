"""Write a GDS with all cells."""

import gdsfactory as gf
from gdsfactory.gpdk import PDK

cells = gf.c

if __name__ == "__main__":
    PDK.activate()

    c1 = cells.straight(cross_section="rib", length=5)
    c2 = cells.straight(cross_section="nitride", length=5)
    c3 = cells.straight(cross_section="strip", length=5)
    c4 = cells.straight(cross_section="pin", length=5)

    c = gf.grid([c1, c2, c3])
    s = c.to_3d()

"""Write a GDS with all cells."""

import gdsfactory as gf
from gdsfactory.gpdk import PDK

cells = gf.c

if __name__ == "__main__":
    PDK.activate()

    c0 = cells.straight_heater_meander()
    c1 = cells.via_stack_m1_mtop()
    c2 = cells.via_stack_heater_mtop()
    c = gf.grid([c0, c1, c2])
    c.show()
    # s = c.to_3d()
    # s.show()

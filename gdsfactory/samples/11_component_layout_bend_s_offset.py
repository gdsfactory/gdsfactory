"""Lets try the bend_s_offset with different p parameters and see how the layout changes."""

import gdsfactory as gf
from gdsfactory.gpdk import PDK

if __name__ == "__main__":
    PDK.activate()
    c = gf.Component()
    with_arc_floorplan = True
    c1 = c << gf.c.bend_s_offset(
        offset=20, p=0.5, with_arc_floorplan=with_arc_floorplan
    )
    c2 = c << gf.c.bend_s_offset(
        offset=20, p=0.0, with_arc_floorplan=with_arc_floorplan
    )
    c3 = c << gf.c.bend_s_offset(
        offset=20, p=1.0, with_arc_floorplan=with_arc_floorplan
    )
    c.show()

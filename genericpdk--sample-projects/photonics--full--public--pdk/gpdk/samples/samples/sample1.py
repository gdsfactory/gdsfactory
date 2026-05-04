"""Write GDS with sample connections."""

import gdsfactory as gf

from gpdk import components as cells


@gf.cell
def sample1_connect() -> gf.Component:
    c = gf.Component()
    wg1 = c << cells.straight(length=1, width=1)
    wg2 = c << cells.straight(length=2, width=1)
    wg3 = c << cells.straight(length=3, width=1)

    wg2.connect(port="o1", other=wg1["o2"])
    wg3.connect(port="o1", other=wg2["o2"])
    return c

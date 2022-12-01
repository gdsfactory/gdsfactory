"""Lets create a new component.

We create a function which returns a gf.Component.

Lets build straight crossing out of a vertical and horizontal arm

- Create a component using a function with the cell decorator to define the name automatically and uniquely.
- Define the polygons in the component
- Add ports to the component so you can connect it with other components

"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory import LAYER
from gdsfactory.component import Component


@gf.cell
def test_crossing_arm(
    wg_width: float = 0.5,
    r1: float = 3.0,
    r2: float = 1.1,
    taper_width: float = 1.2,
    taper_length: float = 3.4,
) -> Component:
    """Returns a crossing arm.

    Args:
        wg_width:
        r1:
        r2:
        taper_width:
        taper_length:

    """
    c = gf.Component()
    c << gf.components.ellipse(radii=(r1, r2), layer=LAYER.SLAB150)

    xmax = taper_length + taper_width / 2
    h = wg_width / 2
    taper_points = [
        (-xmax, h),
        (-taper_width / 2, taper_width / 2),
        (taper_width / 2, taper_width / 2),
        (xmax, h),
        (xmax, -h),
        (taper_width / 2, -taper_width / 2),
        (-taper_width / 2, -taper_width / 2),
        (-xmax, -h),
    ]

    c.add_polygon(taper_points, layer=LAYER.WG)

    c.add_port(
        name="o1", center=(-xmax, 0), orientation=180, width=wg_width, layer=LAYER.WG
    )

    c.add_port(
        name="o2", center=(xmax, 0), orientation=0, width=wg_width, layer=LAYER.WG
    )
    return c


if __name__ == "__main__":
    c = test_crossing_arm()
    c.show(show_ports=True)

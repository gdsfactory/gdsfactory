from typing import Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def nxn(
    west: int = 1,
    east: int = 4,
    north: int = 0,
    south: int = 0,
    xsize: float = 8.0,
    ysize: float = 8.0,
    wg_width: float = 0.5,
    layer: Tuple[int, int] = gf.LAYER.WG,
    wg_margin: float = 1.0,
    **kwargs,
) -> Component:
    """returns a nxn component with nxn ports (west, east, north, south)

    Args:
        west: number of west ports
        east: number of east ports
        north: number of north ports
        south: number of south ports
        xsize: size in X
        ysize: size in Y
        wg_width: width of the straight ports
        wg_margin: margin from straight to component edge
        **kwargs: port_settings


    .. code::

            3   4
            |___|_
        2 -|      |- 5
           |      |
        1 -|______|- 6
            |   |
            8   7

    """
    c = gf.Component()
    c << gf.components.rectangle(size=(xsize, ysize), layer=layer)

    if west > 0:
        x = 0
        y = (
            [ysize / 2]
            if west == 1
            else np.linspace(
                wg_margin + wg_width / 2, ysize - wg_margin - wg_width / 2, west
            )
        )
        y = gf.snap.snap_to_grid(y)
        orientation = 180

        for i, y in enumerate(y):
            c.add_port(
                f"W{i}",
                midpoint=(x, y),
                width=wg_width,
                orientation=orientation,
                layer=layer,
                **kwargs,
            )

    if east > 0:
        x = xsize
        y = (
            [ysize / 2]
            if east == 1
            else np.linspace(
                wg_margin + wg_width / 2, ysize - wg_margin - wg_width / 2, east
            )
        )
        y = gf.snap.snap_to_grid(y)
        orientation = 0

        for i, y in enumerate(y):
            c.add_port(
                f"E{i}",
                midpoint=(x, y),
                width=wg_width,
                orientation=orientation,
                layer=layer,
                **kwargs,
            )

    if north > 0:
        y = ysize
        x = (
            [xsize / 2]
            if north == 1
            else np.linspace(
                wg_margin + wg_width / 2, xsize - wg_margin - wg_width / 2, north
            )
        )
        x = gf.snap.snap_to_grid(x)
        orientation = 90

        for i, x in enumerate(x):
            c.add_port(
                f"N{i}",
                midpoint=(x, y),
                width=wg_width,
                orientation=orientation,
                layer=layer,
                **kwargs,
            )
    if south > 0:
        y = 0
        x = (
            [xsize / 2]
            if south == 1
            else np.linspace(
                wg_margin + wg_width / 2, xsize - wg_margin - wg_width / 2, south
            )
        )
        x = gf.snap.snap_to_grid(x)
        orientation = 270

        for i, x in enumerate(x):
            c.add_port(
                f"S{i}",
                midpoint=(x, y),
                width=wg_width,
                orientation=orientation,
                layer=layer,
                **kwargs,
            )

    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = nxn(north=1.3, south=3)
    # c = gf.components.extension.extend_ports(component=c)
    c.show()

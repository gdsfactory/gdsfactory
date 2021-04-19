from typing import Tuple, Union

import numpy as np
from omegaconf.listconfig import ListConfig

import pp
from pp.component import Component
from pp.port import deco_rename_ports


@deco_rename_ports
@pp.cell
def nxn(
    west: int = 1,
    east: int = 4,
    north: int = 0,
    south: int = 0,
    xsize: float = 8.0,
    ysize: float = 8.0,
    wg_width: float = 0.5,
    layer: Union[ListConfig, Tuple[int, int]] = pp.LAYER.WG,
    wg_margin: float = 1.0,
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

    .. code::

             N0  N1
             |___|_
        W1 -|      |- E1
            |      |
        W0 -|______|- E0
             |   |
            S0   S1

    """
    c = pp.Component()
    c << pp.components.rectangle(size=(xsize, ysize), layer=layer)

    for i in [west, north, south, east]:
        if not isinstance(i, int):
            raise ValueError(
                f"west={west}, east={east}, north={north}, south={south} needs to be integers"
            )

    if west > 0:
        x = 0
        y = (
            [ysize / 2]
            if west == 1
            else np.linspace(
                wg_margin + wg_width / 2, ysize - wg_margin - wg_width / 2, west
            )
        )
        y = pp.snap_to_grid(y)
        orientation = 180

        for i, y in enumerate(y):
            c.add_port(
                f"W{i}",
                midpoint=(x, y),
                width=wg_width,
                orientation=orientation,
                layer=layer,
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
        y = pp.snap_to_grid(y)
        orientation = 0

        for i, y in enumerate(y):
            c.add_port(
                f"E{i}",
                midpoint=(x, y),
                width=wg_width,
                orientation=orientation,
                layer=layer,
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
        x = pp.snap_to_grid(x)
        orientation = 90

        for i, x in enumerate(x):
            c.add_port(
                f"N{i}",
                midpoint=(x, y),
                width=wg_width,
                orientation=orientation,
                layer=layer,
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
        x = pp.snap_to_grid(x)
        orientation = 270

        for i, x in enumerate(x):
            c.add_port(
                f"S{i}",
                midpoint=(x, y),
                width=wg_width,
                orientation=orientation,
                layer=layer,
            )

    return c


if __name__ == "__main__":
    c = nxn(north=1, south=3)
    c = pp.extend_ports(c)
    c.show()

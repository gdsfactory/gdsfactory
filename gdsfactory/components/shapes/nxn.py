from __future__ import annotations

from typing import Any

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def nxn(
    west: int = 1,
    east: int = 4,
    north: int = 0,
    south: int = 0,
    xsize: float = 8.0,
    ysize: float = 8.0,
    wg_width: float = 0.5,
    layer: LayerSpec = "WG",
    wg_margin: float = 1.0,
    **kwargs: Any,
) -> Component:
    """Returns a nxn component with nxn ports (west, east, north, south).

    Args:
        west: number of west ports.
        east: number of east ports.
        north: number of north ports.
        south: number of south ports.
        xsize: size in X.
        ysize: size in Y.
        wg_width: width of the straight ports.
        layer: layer.
        wg_margin: margin from straight to component edge.
        kwargs: port_settings.

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
    _ = c << gf.components.rectangle(size=(xsize, ysize), layer=layer)

    if west > 0:
        x_west = 0
        y_west = (
            [ysize / 2]
            if west == 1
            else list(
                np.linspace(
                    wg_margin + wg_width / 2, ysize - wg_margin - wg_width / 2, west
                )
            )
        )
        orientation = 180

        for i, yi in enumerate(y_west):
            c.add_port(
                f"W{i}",
                center=(float(x_west), float(yi)),
                width=wg_width,
                orientation=orientation,
                layer=layer,
                **kwargs,
            )

    if east > 0:
        x_east = xsize
        y_east = (
            [ysize / 2]
            if east == 1
            else list(
                np.linspace(
                    wg_margin + wg_width / 2, ysize - wg_margin - wg_width / 2, east
                )
            )
        )
        orientation = 0

        for i, yi in enumerate(y_east):
            c.add_port(
                f"E{i}",
                center=(float(x_east), float(yi)),
                width=wg_width,
                orientation=orientation,
                layer=layer,
                **kwargs,
            )

    if north > 0:
        y_north = ysize
        x_north = (
            [xsize / 2]
            if north == 1
            else list(
                np.linspace(
                    wg_margin + wg_width / 2, xsize - wg_margin - wg_width / 2, north
                )
            )
        )
        orientation = 90

        for i, xi in enumerate(x_north):
            c.add_port(
                f"N{i}",
                center=(float(xi), float(y_north)),
                width=wg_width,
                orientation=orientation,
                layer=layer,
                **kwargs,
            )
    if south > 0:
        y_south = 0
        x_south = (
            [xsize / 2]
            if south == 1
            else list(
                np.linspace(
                    wg_margin + wg_width / 2, xsize - wg_margin - wg_width / 2, south
                )
            )
        )
        orientation = 270

        for i, xi in enumerate(x_south):
            c.add_port(
                f"S{i}",
                center=(float(xi), float(y_south)),
                width=wg_width,
                orientation=orientation,
                layer=layer,
                **kwargs,
            )

    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    # c = nxn(north=1.3, south=3)
    c = nxn()
    c.pprint_ports()
    # c = gf.components.extension.extend_ports(component=c)
    c.show()

from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell
def nxn(
    west: int = 1,
    east: int = 4,
    north: int = 0,
    south: int = 0,
    xsize: float | None = None,
    ysize: float | None = None,
    wg_width: float = 0.5,
    layer: LayerSpec = "WG",
    wg_margin: float = 1.0,
    **kwargs,
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
    wg_pitch = wg_margin + wg_width

    if ysize is None:
        ysize = wg_pitch * max([west, east, 1])
    if xsize is None:
        xsize = wg_pitch * max([north, south, 1])

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
        orientation = 180
        y = gf.snap.snap_to_grid(y)

        for i, yi in enumerate(y):
            c.add_port(
                f"W{i}",
                center=(x, yi),
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
        orientation = 0
        y = gf.snap.snap_to_grid(y)

        for i, yi in enumerate(y):
            c.add_port(
                f"E{i}",
                center=(x, yi),
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
        orientation = 90
        x = gf.snap.snap_to_grid(x)

        for i, xi in enumerate(x):
            c.add_port(
                f"N{i}",
                center=(xi, y),
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
        orientation = 270
        x = gf.snap.snap_to_grid(x)

        for i, xi in enumerate(x):
            c.add_port(
                f"S{i}",
                center=(xi, y),
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
    c.show(show_ports=True)

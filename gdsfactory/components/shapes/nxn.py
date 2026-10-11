from __future__ import annotations

__all__ = ["nxn"]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
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

    ```text
            3   4
            |___|_
        2 -|      |- 5
           |      |
        1 -|______|- 6
            |   |
            8   7
    ```
    """
    return cf.nxn(
        west=west,
        east=east,
        north=north,
        south=south,
        xsize=xsize,
        ysize=ysize,
        wg_width=wg_width,
        layer=layer,
        wg_margin=wg_margin,
        **kwargs,
    )

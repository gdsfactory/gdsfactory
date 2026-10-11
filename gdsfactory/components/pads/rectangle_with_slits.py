from __future__ import annotations

__all__ = ["rectangle_with_slits"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec, Size


@gf.cell_with_module_name(tags=["pads"])
def rectangle_with_slits(
    size: Size = (100.0, 200.0),
    layer: LayerSpec = "WG",
    layer_slit: LayerSpec | None = None,
    centered: bool = False,
    port_type: str | None = None,
    slit_size: Size = (1.0, 1.0),
    slit_column_pitch: float = 20,
    slit_row_pitch: float = 20,
    slit_enclosure: float = 10,
) -> Component:
    """Returns a rectangle with slits.

    Metal slits reduce stress.

    Args:
        size: (tuple) Width and height of rectangle.
        layer: Specific layer to put polygon geometry on.
        layer_slit: does a boolean NOT when None.
        centered: True sets center to (0, 0), False sets south-west to (0, 0)
        port_type: for the rectangle.
        slit_size: x, y slit size.
        slit_column_pitch: pitch for columns of slits.
        slit_row_pitch: pitch for rows of slits.
        slit_enclosure: from slit to rectangle edge.

    ```text
        slit_enclosure
        _____________________________________
        |<--->                              |
        |                                   |
        |      ______________________       |
        |     |                      |      |
        |     |                      | slit_size[1]
        |  _  |______________________|      |
        |  |                                |
        |  | slit_row_pitch                 |
        |  |                                |  size[1]
        |  |   ______________________       |
        |  |  |                      |      |
        |  |  |                      |      |
        |  _  |______________________|      |
        |     <--------------------->       |
        |            slit_size[0]           |
        |___________________________________|
                        size[0]
    ```
    """
    return cf.rectangle_with_slits(
        size=size,
        layer=layer,
        layer_slit=layer_slit,
        centered=centered,
        port_type=port_type,
        slit_size=slit_size,
        slit_column_pitch=slit_column_pitch,
        slit_row_pitch=slit_row_pitch,
        slit_enclosure=slit_enclosure,
    )

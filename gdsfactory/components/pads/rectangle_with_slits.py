from __future__ import annotations

__all__ = ["rectangle_with_slits"]

import numpy as np

import gdsfactory as gf
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

    .. code::

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
    """
    c = Component()
    layer_tuple = gf.get_layer_tuple(layer)

    rectangle = gf.c.rectangle(
        size=size, layer=layer, port_type=port_type, centered=centered
    )
    r = c << rectangle
    c.add_ports(r.ports)
    columns = int(np.floor((size[0] - 2 * slit_enclosure) / slit_column_pitch))
    rows = int(np.floor((size[1] - 2 * slit_enclosure) / slit_row_pitch))

    if layer_slit is None:
        layer2 = (layer_tuple[0], layer_tuple[1] + 1)
        slit = gf.c.rectangle(size=slit_size, port_type=None, layer=layer2)
        slits = gf.c.array(
            slit,
            columns=columns,
            rows=rows,
            column_pitch=slit_column_pitch,
            row_pitch=slit_row_pitch,
            centered=centered,
        )
        slits_ref = c << slits
        slits_ref.center = r.center
        c = gf.boolean(
            rectangle,
            slits_ref,
            operation="not",
            layer1=layer,
            layer2=layer2,
            layer=layer,
        )
        c.add_ports(rectangle.ports)

    else:
        slit = gf.c.rectangle(size=slit_size, port_type=None, layer=layer_slit)
        slits_ref = c << gf.c.array(
            slit,
            columns=columns,
            rows=rows,
            column_pitch=slit_column_pitch,
            row_pitch=slit_row_pitch,
            centered=centered,
        )
        slits_ref.center = r.center
    return c

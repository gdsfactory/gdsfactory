from __future__ import annotations

import warnings

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.array_component import array
from gdsfactory.components.rectangle import rectangle
from gdsfactory.typings import Float2, LayerSpec, Size


@gf.cell
def rectangle_with_slits(
    size: Size = (100.0, 200.0),
    layer: LayerSpec = "WG",
    layer_slit: LayerSpec = "SLAB150",
    centered: bool = False,
    port_type: str | None = None,
    slit_size: Size = (1.0, 1.0),
    slit_spacing: Float2 | None = None,
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
        slit_spacing: pitch_x, pitch_y for slits.
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
    layer = gf.get_layer(layer)

    if slit_spacing:
        warnings.warn(
            "slit_spacing is deprecated. Use slit_column_pitch and slit_row_pitch instead"
        )
        slit_column_pitch = slit_spacing[0]
        slit_row_pitch = slit_spacing[1]

    r = c << rectangle(size=size, layer=layer, port_type=port_type, centered=centered)
    c.add_ports(r.ports)
    slit = rectangle(size=slit_size, port_type=None, layer=layer_slit or layer)
    columns = np.floor((size[0] - 2 * slit_enclosure) / slit_column_pitch)
    rows = np.floor((size[1] - 2 * slit_enclosure) / slit_row_pitch)
    slits = c << array(
        slit,
        columns=columns,
        rows=rows,
        spacing=slit_spacing,
        column_pitch=slit_column_pitch,
        row_pitch=slit_row_pitch,
    )
    slits.dcenter = r.dcenter
    return c


if __name__ == "__main__":
    # c = rectangle_with_slits(layer_slit=None)
    c = rectangle_with_slits(slit_size=(10, 10), centered=True)
    c.show()

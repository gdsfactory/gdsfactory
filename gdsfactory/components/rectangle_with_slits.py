from typing import Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.array import array
from gdsfactory.components.rectangle import rectangle
from gdsfactory.types import Float2, Layer


@cell
def rectangle_with_slits(
    size: Tuple[float, float] = (100.0, 200.0),
    layer: Layer = (1, 0),
    centered: bool = False,
    port_type: Optional[str] = None,
    slit_size: Tuple[float, float] = (1.0, 1.0),
    slit_spacing: Float2 = (20, 20),
    slit_enclosure: float = 10,
) -> Component:
    """Returns a rectangle with slits. Metal slits reduce stress.

    Args:
        size: (tuple) Width and height of rectangle.
        layer: Specific layer to put polygon geometry on.
        centered: True sets center to (0, 0), False sets south-west to (0, 0)
        port_type: for the rectangle
        slit_size: x, y slit size
        slit_spacing: pitch_x, pitch_y for slits
        slit_enclosure: from slit to rectangle edge


    .. code::

        slit_enclosure
        _____________________________________
        |<--->                              |
        |                                   |
        |      ______________________       |
        |     |                      |      |
        |     |                      | slit_size[1]
        |     |______________________|      |
        |  |                                |
        |  | slit_spacing                   |
        |  |                                |  size[1]
        |  |   ______________________       |
        |  |  |                      |      |
        |  |  |                      |      |
        |  |  |______________________|      |
        |     <--------------------->       |
        |            slit_size[0]           |
        |___________________________________|
                        size[0]


    """
    c = Component()
    r = rectangle(size=size, layer=layer, port_type=port_type, centered=centered)
    c.add_ports(r.ports)
    slit = rectangle(size=slit_size, port_type=None, layer=layer)

    columns = np.floor(size[0] / slit_spacing[0])
    rows = np.floor(size[1] / slit_spacing[1])
    slits = array(slit, columns=columns, rows=rows, spacing=slit_spacing).ref()
    slits.xmin = slit_enclosure
    slits.ymin = slit_enclosure

    r_with_slits = c << gf.geometry.boolean(r, slits, operation="not", layer=layer)
    c.absorb(r_with_slits)
    return c


if __name__ == "__main__":
    c = rectangle_with_slits()
    c.show()

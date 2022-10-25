from typing import Optional, Tuple, Union

import numpy as np

import gdsfactory as gf
from gdsfactory.components.text import text
from gdsfactory.types import Anchor, LayerSpec

Coordinate = Union[Tuple[float, float], Tuple[int, int]]


@gf.cell_without_validator
def die_bbox_frame(
    bbox: Tuple[Coordinate, Coordinate] = ((-1.0, -1.0), (3.0, 4.0)),
    street_width: float = 100.0,
    street_length: float = 1000.0,
    die_name: Optional[str] = None,
    text_size: float = 100.0,
    text_anchor: Anchor = "sw",
    layer: LayerSpec = "M3",
    padding: float = 10.0,
) -> gf.Component:
    """Return boundary box frame.

    Perfect for defining dicing lanes. the
    boundary of the chip/die it can also add a label with the name of the die.
    similar to die and bbox.

    based on phidl.geometry

    Args:
        bbox: bounding box to frame.
        street_width: Width of the boundary box.
        street_length: length of the boundary box.
        die_name: Label text.
        text_size: Label text size.
        text_anchor: {'nw', 'nc', 'ne', 'sw', 'sc', 'se'} text location.
        layer: Specific layer(s) to put polygon geometry on.
        padding: adds padding.
    """
    D = gf.Component()
    (xmin, ymin), (xmax, ymax) = bbox

    x = (xmax + xmin) / 2
    y = (ymax + ymin) / 2

    sx = xmax - xmin
    sy = ymax - ymin

    sx = sx / 2
    sy = sy / 2

    sx += street_width + padding
    sy += street_width + padding
    street_length = max([sx, sy])

    xpts = np.array(
        [
            sx,
            sx,
            sx - street_width,
            sx - street_width,
            sx - street_length,
            sx - street_length,
        ]
    )
    ypts = np.array(
        [
            sy,
            sy - street_length,
            sy - street_length,
            sy - street_width,
            sy - street_width,
            sy,
        ]
    )
    D.add_polygon([+xpts, +ypts], layer=layer)
    D.add_polygon([-xpts, +ypts], layer=layer)
    D.add_polygon([+xpts, -ypts], layer=layer)
    D.add_polygon([-xpts, -ypts], layer=layer)

    if die_name:
        t = D.add_ref(text(text=die_name, size=text_size, layer=layer))
        d = street_width + 20

        if text_anchor == "nw":
            t.xmin, t.ymax = [-sx + d, sy - d]
        elif text_anchor == "nc":
            t.x, t.ymax = [0, sy - d]
        elif text_anchor == "ne":
            t.xmax, t.ymax = [sx - d, sy - d]
        if text_anchor == "sw":
            t.xmin, t.ymin = [-sx + d, -sy + d]
        elif text_anchor == "sc":
            t.x, t.ymin = [0, -sy + d]
        elif text_anchor == "se":
            t.xmax, t.ymin = [sx - d, -sy + d]

    return D.move((x, y)).flatten()


if __name__ == "__main__":
    c = gf.Component("demo")
    mask = c << gf.components.array(rows=15, columns=10)
    c << die_bbox_frame(mask.bbox, die_name="chip99")
    c.show(show_ports=True)

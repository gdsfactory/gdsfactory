from typing import Optional

import numpy as np

import gdsfactory as gf
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.text import text
from gdsfactory.types import Layer

big_square = gf.partial(rectangle, size=(1300, 2600))


@gf.cell
def die_bbox(
    component: gf.types.ComponentOrFactory = big_square,
    street_width: float = 100.0,
    street_length: float = 1000.0,
    die_name: Optional[str] = None,
    text_size: float = 100.0,
    text_location: str = "SW",
    layer: Layer = gf.LAYER.FLOORPLAN,
    padding: float = 10.0,
) -> gf.Component:
    """Creates a basic boundary box. Perfect for defining dicing lanes.
    the boundary of the chip/die
    it can also add a label with the name of the die.
    similar to die and bbox

    adapted from phidl.geometry

    Args:
        component: to frame
        street_width: Width of the boundary box
        die_name: Label text.
        text_size: Label text size.
        text_location: {'NW', 'N', 'NE', 'SW', 'S', 'SE'} Label text compass location.
        layer: Specific layer(s) to put polygon geometry on.
        padding: adds padding

    """
    D = gf.Component()
    component = component() if callable(component) else component
    cref = D.add_ref(component)
    cref.x = 0
    cref.y = 0
    size = cref.size
    sx, sy = size[0] / 2, size[1] / 2

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
        if type(text_location) is str:
            text_location = text_location.upper()
            if text_location == "NW":
                t.xmin, t.ymax = [-sx + d, sy - d]
            elif text_location == "N":
                t.x, t.ymax = [0, sy - d]
            elif text_location == "NE":
                t.xmax, t.ymax = [sx - d, sy - d]
            if text_location == "SW":
                t.xmin, t.ymin = [-sx + d, -sy + d]
            elif text_location == "S":
                t.x, t.ymin = [0, -sy + d]
            elif text_location == "SE":
                t.xmax, t.ymin = [sx - d, -sy + d]
        else:
            t.x, t.y = text_location

    return D


if __name__ == "__main__":
    mask = gf.components.array_2d(rows=10, cols=10)
    c = die_bbox(component=mask)
    # c = die_bbox()
    c.show()

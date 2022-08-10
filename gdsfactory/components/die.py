"""Adapted from phidl.geometry."""

from typing import Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.components.text import text
from gdsfactory.types import LayerSpec


@gf.cell
def die(
    size: Tuple[float, float] = (10000.0, 10000.0),
    street_width: float = 100.0,
    street_length: float = 1000.0,
    die_name: Optional[str] = "chip99",
    text_size: float = 100.0,
    text_location: str = "SW",
    layer: LayerSpec = "FLOORPLAN",
    bbox_layer: Optional[LayerSpec] = "FLOORPLAN",
    draw_corners: bool = True,
    draw_dicing_lane: bool = True,
) -> gf.Component:
    """Returns basic die with 4 right angle corners marking the boundary of the.

    chip/die and a label with the name of the die.

    Args:
        size: x, y dimensions of the die.
        street_width: Width of the corner marks for die-sawing.
        street_length: Length of the corner marks for die-sawing.
        die_name: Label text.
        text_size: Label text size.
        text_location: {'NW', 'N', 'NE', 'SW', 'S', 'SE'} Label text compass location.
        layer: Specific layer to put polygon geometry on.
        bbox_layer: optional bbox layer.
        draw_corners: around die.
        draw_dicing_lane: around die.
    """
    c = gf.Component(name="die")
    sx, sy = size[0] / 2, size[1] / 2

    if draw_dicing_lane:
        street_length = max([sx, sy])

    if draw_corners or draw_dicing_lane:
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
        c.add_polygon([xpts, ypts], layer=layer)
        c.add_polygon([-xpts, ypts], layer=layer)
        c.add_polygon([xpts, -ypts], layer=layer)
        c.add_polygon([-xpts, -ypts], layer=layer)

    if bbox_layer:
        c.add_polygon([[sx, sy], [sx, -sy], [-sx, -sy], [-sx, sy]], layer=bbox_layer)

    if die_name:
        t = c.add_ref(text(text=die_name, size=text_size, layer=layer))

        d = street_width + 20
        if type(text_location) is str:
            text_location = text_location.upper()
            if text_location == "N":
                t.x, t.ymax = [0, sy - d]
            elif text_location == "NE":
                t.xmax, t.ymax = [sx - d, sy - d]
            elif text_location == "NW":
                t.xmin, t.ymax = [-sx + d, sy - d]
            elif text_location == "S":
                t.x, t.ymin = [0, -sy + d]
            elif text_location == "SE":
                t.xmax, t.ymin = [sx - d, -sy + d]
            elif text_location == "SW":
                t.xmin, t.ymin = [-sx + d, -sy + d]
        else:
            t.x, t.y = text_location

    return c


if __name__ == "__main__":
    # c = die(size=(3000, 5000), draw_dicing_lane=True)
    c = die()
    # c.show(show_ports=True)
    c.plot()
    c.show()

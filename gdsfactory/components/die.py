"""based on phidl.geometry."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

import gdsfactory as gf
from gdsfactory.components.text import text
from gdsfactory.typings import LayerSpec, ComponentSpec, Float2


@gf.cell
def die(
    size: Tuple[float, float] = (10000.0, 10000.0),
    street_width: float = 100.0,
    street_length: float = 1000.0,
    die_name: Optional[str] = "chip99",
    text_size: float = 100.0,
    text_location: Union[str, Float2] = "SW",
    layer: LayerSpec = "FLOORPLAN",
    bbox_layer: Optional[LayerSpec] = "FLOORPLAN",
    draw_corners: bool = True,
    draw_dicing_lane: bool = True,
    text_component: ComponentSpec = text,
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
        text_component: component to use for generating text
    """
    c = gf.Component(name="die")
    sx, sy = size[0] / 2, size[1] / 2

    if draw_dicing_lane:
        street_length = sy

    if draw_corners or draw_dicing_lane:
        xpts = np.array([sx, sx, sx - street_width, sx - street_width, 0, 0])
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
        t = c.add_ref(text_component(text=die_name, size=text_size, layer=layer))

        d = street_width + 20
        if isinstance(text_location, str):
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
    # c = die()
    c = gf.components.die(
        size=(13000, 3000),  # Size of die
        street_width=100,  # Width of corner marks for die-sawing
        street_length=1000,  # Length of corner marks for die-sawing
        die_name="chip99",  # Label text
        text_size=500,  # Label text size
        text_location="SW",  # Label text compass location e.g. 'S', 'SE', 'SW'
        layer=(2, 0),
        bbox_layer=(3, 0),
    )
    c.show()
    # c.show(show_ports=True)
    # c.plot()

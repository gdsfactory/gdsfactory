"""based on phidl.geometry."""

from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.components.text import text
from gdsfactory.typings import ComponentFactory, Float2, LayerSpec


@gf.cell
def die(
    size: tuple[float, float] = (10000.0, 10000.0),
    street_width: float = 100.0,
    street_length: float = 1000.0,
    die_name: str | None = "chip99",
    text_size: float = 100.0,
    text_location: str | Float2 = "SW",
    layer: LayerSpec | None = "FLOORPLAN",
    bbox_layer: LayerSpec | None = "FLOORPLAN",
    text: ComponentFactory = text,
    draw_corners: bool = False,
) -> gf.Component:
    """Returns die with optional markers marking the boundary of the die.

    Args:
        size: x, y dimensions of the die.
        street_width: Width of the corner marks for die-sawing.
        street_length: Length of the corner marks for die-sawing.
        die_name: Label text. If None, no label is added.
        text_size: Label text size.
        text_location: {'NW', 'N', 'NE', 'SW', 'S', 'SE'} or (x, y) coordinate.
        layer: For street widths. None to not draw the street widths.
        bbox_layer: optional bbox layer drawn bounding box around the die.
        text: function use for generating text. Needs to accept text, size, layer.
        draw_corners: True draws only corners. False draws a square die.
    """
    c = gf.Component()
    sx, sy = size[0] / 2, size[1] / 2

    if layer:
        if not draw_corners:
            street_length = sx
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
        if not draw_corners:
            street_length = sy
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
                raise ValueError(
                    f"Invalid text_location: {text_location} not in N, NE, NW, S, SE, SW"
                )
        else:
            t.x, t.y = text_location

    return c


if __name__ == "__main__":
    # c = die(size=(3000, 5000), draw_dicing_lane=True)
    # c = die()
    c = die(
        size=(13000, 3000),  # Size of die
        street_width=100,  # Width of corner marks for die-sawing
        street_length=1000,  # Length of corner marks for die-sawing
        die_name="chip99",  # Label text
        text_size=500,  # Label text size
        text_location="SW",  # Label text compass location e.g. 'S', 'SE', 'SW'
        layer=(2, 0),
        # bbox_layer=(3, 0),
        # bbox_layer=None,
    )
    c.show()
    # c.show(show_ports=True)
    # c.plot()

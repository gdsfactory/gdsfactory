"""based on phidl.geometry."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.components_functions import die as die_function
from gdsfactory.typings import ComponentSpec, Float2, LayerSpec, Size


@gf.cell_with_module_name
def die(
    size: Size = (10000.0, 10000.0),
    street_width: float = 100.0,
    street_length: float = 1000.0,
    die_name: str | None = "chip99",
    text_size: float = 100.0,
    text_location: str | Float2 = "SW",
    layer: LayerSpec | None = "FLOORPLAN",
    bbox_layer: LayerSpec | None = "FLOORPLAN",
    text: ComponentSpec = "text",
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
    return die_function(
        size=size,
        street_width=street_width,
        street_length=street_length,
        die_name=die_name,
        text_size=text_size,
        text_location=text_location,
        layer=layer,
        bbox_layer=bbox_layer,
        text=text,
        draw_corners=draw_corners,
    )


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
    # c.show( )
    # c.plot()

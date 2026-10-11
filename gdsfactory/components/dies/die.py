"""based on phidl.geometry."""

from __future__ import annotations

__all__ = ["die"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.typings import ComponentSpec, Float2, LayerSpec, Size


@gf.cell_with_module_name(tags=["dies"])
def die(
    size: Size = (10000.0, 10000.0),
    street_width: float = 100.0,
    street_length: float = 1000.0,
    die_name: str | None = "chip99",
    text_size: float = 100.0,
    text_location: str | Float2 = "SW",
    layer: LayerSpec | None = "FLOORPLAN",
    bbox_layer: LayerSpec | None = "FLOORPLAN",
    text_layer: LayerSpec = "WG",
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
        text_layer: Layer for the die name text.
        text: function use for generating text. Needs to accept text, size, layer.
        draw_corners: True draws only corners. False draws a square die.
    """
    return cf.die(
        size=size,
        street_width=street_width,
        street_length=street_length,
        die_name=die_name,
        text_size=text_size,
        text_location=text_location,
        layer=layer,
        bbox_layer=bbox_layer,
        text_layer=text_layer,
        text=text,
        draw_corners=draw_corners,
    )

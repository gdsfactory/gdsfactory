from __future__ import annotations

__all__ = ["character_a", "pixel_array", "rectangular_font"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions.texts.text_rectangular_font import (
    character_a,
    rectangular_font,
)
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["texts"])
def pixel_array(
    pixels: str = character_a,
    pixel_size: float = 10.0,
    layer: LayerSpec = "M1",
) -> Component:
    """Returns a pixel component from a string representing the pixels.

    Args:
        pixels: string representing the pixels
        pixel_size: width/height for each pixel
        layer: layer for each pixel
    """
    return cf.pixel_array(pixels=pixels, pixel_size=pixel_size, layer=layer)

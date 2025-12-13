from __future__ import annotations

__all__ = ["text_rectangular", "text_rectangular_multi_layer"]

from collections.abc import Callable
from functools import partial
from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs

from ..containers.copy_layers import copy_layers
from ..texts.text_rectangular_font import (
    pixel_array,
    rectangular_font,
)


@gf.cell_with_module_name
def text_rectangular(
    text: str = "abcd",
    size: float = 10.0,
    position: tuple[float, float] = (0.0, 0.0),
    justify: str = "left",
    layer: LayerSpec | None = "WG",
    layers: LayerSpecs | None = None,
    font: Callable[..., dict[str, str]] = rectangular_font,
) -> Component:
    """Pixel based font, guaranteed to be manhattan, without acute angles.

    Args:
        text: string.
        size: pixel size in um.
        position: coordinate.
        justify: left, right or center.
        layer: for text.
        layers: optional for duplicating the text.
        font: function that returns dictionary of characters.
    """
    pixel_size = size
    xoffset = position[0]
    yoffset = position[1]
    component = gf.Component()
    characters = font()

    if layers is None:
        assert layer is not None, "layer is None. Please provide a layer."
        layers = [layer]

    # Extract pixel width count from font definition.
    # Example below is 5, and 7 for FONT_LITHO.
    # A: 1 1 1 1 1
    pixel_width_count = len(characters["A"].split("\n")[0])

    xoffset_factor = pixel_width_count + 1

    for line in text.split("\n"):
        for character in line:
            if character == " ":
                xoffset += pixel_size * xoffset_factor
            elif character.upper() not in characters:
                print(f"skipping character {character!r} not in font")
            else:
                pixels = characters[character.upper()]
                for layer in layers:
                    ref = component.add_ref(
                        pixel_array(pixels=pixels, pixel_size=pixel_size, layer=layer)
                    )
                    ref.move((xoffset, yoffset))
                    component.absorb(ref)
                xoffset += pixel_size * xoffset_factor

        yoffset -= pixel_size * xoffset_factor
        xoffset = position[0]

    c = gf.Component()
    ref = c << component
    justify = justify.lower()
    if justify == "left":
        ref.xmin = position[0]
    elif justify == "right":
        ref.xmax = position[0]
    elif justify == "center":
        ref.x = 0
    else:
        raise ValueError(f"{justify=} not valid (left, center, right)")
    c.flatten()
    return c


@gf.cell_with_module_name
def text_rectangular_multi_layer(
    text: str = "abcd",
    layers: LayerSpecs = ("WG", "M1", "M2", "MTOP"),
    text_factory: ComponentSpec = text_rectangular,
    **kwargs: Any,
) -> Component:
    """Returns rectangular text in different layers.

    Args:
        text: string of text.
        layers: list of layers to replicate the text.
        text_factory: function to create the text Components.
        kwargs: keyword arguments for text_factory.

    Keyword Args:
        size: pixel size.
        position: coordinate.
        justify: left, right or center.
        font: function that returns dictionary of characters.
    """
    return copy_layers(factory=text_factory, text=text, layers=layers, **kwargs)


text_rectangular_mini = partial(text_rectangular, size=1)

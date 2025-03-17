from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.containers.copy_layers import copy_layers
from gdsfactory.components.texts.text_rectangular_font import (
    pixel_array,
    rectangular_font,
)
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs


@gf.cell
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
        size: pixel size.
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
    layer_list = layers or [layer] if layer else []

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
                for layer in layer_list:
                    ref = component.add_ref(
                        pixel_array(pixels=pixels, pixel_size=pixel_size, layer=layer)
                    )
                    ref.dmove((xoffset, yoffset))
                    component.absorb(ref)
                xoffset += pixel_size * xoffset_factor

        yoffset -= pixel_size * xoffset_factor
        xoffset = position[0]

    c = gf.Component()
    ref = c << component
    justify = justify.lower()
    if justify == "left":
        ref.dxmin = position[0]
    elif justify == "right":
        ref.dxmax = position[0]
    elif justify == "center":
        ref.dx = 0
    else:
        raise ValueError(f"{justify=} not valid (left, center, right)")
    c.flatten()
    return c


@gf.cell
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

if __name__ == "__main__":
    c = gf.Component()
    text0 = c << gf.components.text_rectangular(
        text="Center", size=10, position=(0, 40), justify="center", layer=(100, 0)
    )
    text1 = c << gf.components.text_rectangular(
        text="Left", size=10, position=(0, 40), justify="left", layer=(100, 0)
    )
    text2 = c << gf.components.text_rectangular(
        text="Right", size=10, position=(0, 40), justify="right", layer=(100, 0)
    )

    text1.ymin = text0.ymax + 10
    text2.ymin = text1.ymax + 10
    c.show()

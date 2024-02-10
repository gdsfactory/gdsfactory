from __future__ import annotations

from collections.abc import Callable
from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.copy_layers import copy_layers
from gdsfactory.components.text_rectangular_font import pixel_array, rectangular_font
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs


@gf.cell
def text_rectangular(
    text: str = "abcd",
    size: float = 10.0,
    position: tuple[float, float] = (0.0, 0.0),
    justify: str = "left",
    layer: LayerSpec = "WG",
    layers: LayerSpecs | None = None,
    font: Callable[..., dict[str, str]] = rectangular_font,
    post_process: Callable | None = None,
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
        post_process: function to post process the component.
    """
    pixel_size = size
    xoffset = position[0]
    yoffset = position[1]
    component = gf.Component()
    characters = font()
    layers = layers or [layer]

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
        pass
    elif justify == "right":
        ref.xmax = position[0]
    elif justify == "center":
        ref.move(origin=ref.center, destination=position, axis="x")
    else:
        raise ValueError(f"justify = {justify!r} not valid (left, center, right)")
    c.absorb(ref)

    if post_process:
        post_process(c)
    return c


@gf.cell
def text_rectangular_multi_layer(
    text: str = "abcd",
    layers: LayerSpecs = ("WG", "M1", "M2", "MTOP"),
    text_factory: ComponentSpec = text_rectangular,
    post_process: Callable | None = None,
    **kwargs,
) -> Component:
    """Returns rectangular text in different layers.

    Args:
        text: string of text.
        layers: list of layers to replicate the text.
        text_factory: function to create the text Components.
        post_process: function to post process the component.

    Keyword Args:
        size: pixel size
        position: coordinate
        justify: left, right or center
        font: function that returns dictionary of characters
    """
    c = gf.Component()
    _ = c << copy_layers(
        factory=partial(text_factory, text=text, **kwargs),
        layers=layers,
    )
    if post_process:
        post_process(c)
    return c


if __name__ == "__main__":
    c = text_rectangular(
        text="The mask is nearly done. only 12345 drc errors remaining?",
        layers=("SLAB90", "M2"),
        justify="center",
    )
    c.show(show_ports=True)

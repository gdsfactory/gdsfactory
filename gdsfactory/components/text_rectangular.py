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
    font: Callable = rectangular_font,
) -> Component:
    """Pixel based font, guaranteed to be manhattan, without acute angles.

    Args:
        text: string.
        size: pixel size.
        position: coordinate.
        justify: left, right or center.
        layer: for text.
        font: function that returns dictionary of characters.
    """
    pixel_size = size
    xoffset = position[0]
    yoffset = position[1]
    component = gf.Component()
    characters = rectangular_font()

    for line in text.split("\n"):
        for character in line:
            if character == " ":
                xoffset += pixel_size * 6
            elif character.upper() not in characters:
                print(f"skipping character {character!r} not in font")
            else:
                pixels = characters[character.upper()]
                ref = component.add_ref(
                    pixel_array(pixels=pixels, pixel_size=pixel_size, layer=layer)
                )
                ref.move((xoffset, yoffset))
                component.absorb(ref)
                xoffset += pixel_size * 6

        yoffset -= pixel_size * 6
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

    return c


def text_rectangular_multi_layer(
    text: str = "abcd",
    layers: LayerSpecs = ("WG", "M1", "M2", "M3"),
    text_factory: ComponentSpec = text_rectangular,
    **kwargs,
) -> Component:
    """Returns rectangular text in different layers.

    Args:
        text: string of text
        layers: list of layers to replicate the text
        text_factory: function to create the text Components

    Keyword Args:
        size: pixel size
        position: coordinate
        justify: left, right or center
        font: function that returns dictionary of characters
    """
    func = partial(
        copy_layers,
        factory=partial(text_factory, text=text, **kwargs),
        layers=layers,
    )
    return func()


if __name__ == "__main__":
    import string

    c = text_rectangular_multi_layer(
        # text="The mask is nearly done. only 12345 drc errors remaining?",
        # text="v",
        text=string.ascii_lowercase,
        layers=("SLAB90", "M2"),
        justify="center",
    )
    c.show(show_ports=True)

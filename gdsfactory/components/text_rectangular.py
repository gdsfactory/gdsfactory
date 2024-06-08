from __future__ import annotations

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
) -> Component:
    """Pixel based font, guaranteed to be manhattan, without acute angles.

    Args:
        text: string.
        size: pixel size.
        position: coordinate.
        justify: left, right or center.
        layer: for text.
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
                ref.dmove((xoffset, yoffset))
                xoffset += pixel_size * 6

        yoffset -= pixel_size * 6
        xoffset = position[0]

    c = gf.Component()
    ref = c << component
    justify = justify.lower()
    if justify == "left":
        pass
    elif justify == "right":
        ref.dxmax = position[0]
    elif justify == "center":
        ref.dmove(origin=ref.dcenter, other=position, axis="x")
    else:
        raise ValueError(f"justify = {justify!r} not valid (left, center, right)")
    c.flatten()
    return c


def text_rectangular_multi_layer(
    text: str = "abcd",
    layers: LayerSpecs = ("WG", "M1", "M2", "MTOP"),
    text_factory: ComponentSpec = text_rectangular,
    **kwargs,
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
    return copy_layers(
        factory=partial(text_factory, text=text, **kwargs),
        layers=layers,
    )


text_rectangular_mini = partial(text_rectangular, size=1)

if __name__ == "__main__":
    c = text_rectangular_multi_layer()
    c.show()

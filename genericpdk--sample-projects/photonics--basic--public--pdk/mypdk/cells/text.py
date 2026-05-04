"""Functions for creating physical text geometries."""

from typing import Any

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs


@gf.cell
def text_rectangular(
    text: str = "abc",
    size: float = 3,
    justify: str = "left",
    layer: LayerSpec = "PAD",
) -> gf.Component:
    """Pixel based font, guaranteed to be manhattan, without acute angles.

    Args:
        text: string.
        size: pixel size.
        justify: left, right or center.
        layer: for text.
    """
    return gf.c.text_rectangular(
        text=text, size=size, justify=justify, position=(0.0, 0.0), layer=layer
    )


@gf.cell
def text_rectangular_multi_layer(
    text: str = "abc",
    layers: LayerSpecs = ("WG", "PAD"),
    text_factory: ComponentSpec = "text_rectangular",
    **kwargs: Any,
) -> gf.Component:
    """Returns rectangular text in different layers.

    Args:
        text: string of text.
        layers: list of layers to replicate the text.
        text_factory: function to create the text Components.
        kwargs: keyword arguments for text_factory.
    """
    return gf.c.text_rectangular_multi_layer(
        text=text,
        layers=layers,
        text_factory=text_factory,
        **kwargs,
    )

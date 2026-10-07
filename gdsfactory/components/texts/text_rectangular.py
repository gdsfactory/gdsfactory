from __future__ import annotations

__all__ = ["text_rectangular", "text_rectangular_multi_layer"]

from collections.abc import Callable
from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.component_functions.texts.text_rectangular_font import (
    rectangular_font,
)
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs


@gf.cell_with_module_name(tags=["texts"])
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
    return cf.text_rectangular(
        text=text,
        size=size,
        position=position,
        justify=justify,
        layer=layer,
        layers=layers,
        font=font,
    )


@gf.cell_with_module_name(tags=["texts"])
def text_rectangular_multi_layer(
    text: str = "abcd",
    layers: LayerSpecs = ("WG", "M1", "M2", "MTOP"),
    text_factory: ComponentSpec = "text_rectangular",
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
    return cf.text_rectangular_multi_layer(
        text=text, layers=layers, text_factory=text_factory, **kwargs
    )


text_rectangular_mini = CellAlias(text_rectangular, size=1)

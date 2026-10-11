from __future__ import annotations

__all__ = ["text", "text_klayout", "text_lines"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import Coordinate, LayerSpec, LayerSpecs


@gf.cell_with_module_name(tags=["texts"])
def text(
    text: str = "abcd",
    size: float = 10.0,
    position: Coordinate = (0, 0),
    justify: str = "left",
    layer: LayerSpec = "WG",
) -> Component:
    """Text shapes.

    Args:
        text: string.
        size: in um of each character.
        position: x, y position.
        justify: left, right, center.
        layer: for the text.
    """
    return cf.text(
        text=text,
        size=size,
        position=position,
        justify=justify,
        layer=layer,
    )


@gf.cell_with_module_name(tags=["texts"])
def text_lines(
    text: tuple[str, ...] = ("Chip", "01"),
    size: float = 0.4,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a Component from a text lines.

    Args:
        text: list of strings.
        size: text size.
        layer: text layer.
    """
    return cf.text_lines(text=text, size=size, layer=layer)


@gf.cell_with_module_name(tags=["texts"])
def text_klayout(
    text: str = "a",
    layer: LayerSpec = "WG",
    layers: LayerSpecs | None = None,
    bbox_layers: LayerSpecs | None = None,
) -> Component:
    """Returns a text component.

    Args:
        text: string.
        layer: text layer.
        layers: layers for the text.
        bbox_layers: layers for the text bounding box.
    """
    return cf.text_klayout(
        text=text,
        layer=layer,
        layers=layers,
        bbox_layers=bbox_layers,
    )

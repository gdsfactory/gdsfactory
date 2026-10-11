from __future__ import annotations

__all__ = ["text_freetype"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.config import PATH
from gdsfactory.typings import LayerSpec, LayerSpecs, PathType


@gf.cell_with_module_name(tags=["texts"])
def text_freetype(
    text: str = "a",
    size: int = 10,
    justify: str = "left",
    font: PathType = PATH.font_ocr,
    layer: LayerSpec = "WG",
    layers: LayerSpecs | None = None,
) -> Component:
    """Returns text Component.

    Args:
        text: string.
        size: in um.
        justify: left, right, center.
        font: Font face to use. Default DEPLOF does not require additional libraries,
            otherwise freetype load fonts. You can choose font by name
            (e.g. "Times New Roman"), or by file OTF or TTF filepath.
        layer: list of layers to use for the text.
        layers: list of layers to use for the text.

    """
    return cf.text_freetype(
        text=text,
        size=size,
        justify=justify,
        font=font,
        layer=layer,
        layers=layers,
    )

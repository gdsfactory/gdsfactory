from __future__ import annotations

import pathlib
import warnings

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.config import PATH
from gdsfactory.constants import _glyph, _indent, _width
from gdsfactory.typings import LayerSpec, LayerSpecs, PathType


@gf.cell
def text_freetype(
    text: str = "abcd",
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
        position: x, y position.
        justify: left, right, center.
        font: Font face to use. Default DEPLOF does not require additional libraries,
            otherwise freetype load fonts. You can choose font by name
            (e.g. "Times New Roman"), or by file OTF or TTF filepath.
        layer: for the text.
        layers: optional list of layers for the text.
    """
    t = Component()
    layers = layers or [layer]
    yoffset = 0
    xoffset = 0

    if font == "DEPLOF":
        scaling = size / 1000

        for line in text.split("\n"):
            char = Component()
            for c in line:
                ascii_val = ord(c)
                if c == " ":
                    xoffset += 500 * scaling
                elif (33 <= ascii_val <= 126) or (ascii_val == 181):
                    for poly in _glyph[ascii_val]:
                        xpts = np.array(poly)[:, 0] * scaling
                        ypts = np.array(poly)[:, 1] * scaling
                        for layer in layers:
                            char.add_polygon(
                                [xpts + xoffset, ypts + yoffset], layer=layer
                            )
                    xoffset += (_width[ascii_val] + _indent[ascii_val]) * scaling
                else:
                    valid_chars = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~µ"
                    warnings.warn(
                        f'text(): Warning, some characters ignored, no geometry for character "{chr(ascii_val)}" with ascii value {ascii_val}. Valid characters: {valid_chars}'
                    )
            ref = t.add_ref(char)
            t.absorb(ref)
            yoffset -= 1500 * scaling
            xoffset = 0
    else:
        from gdsfactory.font import _get_font_by_file, _get_font_by_name, _get_glyph

        font_path = pathlib.Path(font)

        # Load the font. If we've passed a valid file, try to load that, otherwise search system fonts
        if font_path.is_file() and font_path.suffix in (".otf", ".ttf"):
            font = _get_font_by_file(str(font))
        else:
            font = _get_font_by_name(font)
        if font is None:
            raise ValueError(
                f"Failed to find font: {font!r}. "
                "Try specifying the exact (full) path to the .ttf or .otf file. "
            )

        # Render each character
        for line in text.split("\n"):
            char = Component()
            xoffset = 0
            for letter in line:
                letter_dev = Component()
                letter_template, advance_x = _get_glyph(font, letter)
                for poly in letter_template.polygons:
                    for layer in layers:
                        letter_dev.add_polygon(poly, layer=layer)
                ref = char.add_ref(letter_dev)
                ref.move(destination=(xoffset, 0))
                ref.magnification = size
                xoffset += size * advance_x

            ref = t.add_ref(char)
            ref.move(destination=(0, yoffset))
            yoffset -= size
            t.absorb(ref)

    justify = justify.lower()
    for ref in t.references:
        if justify == "center":
            ref.move(origin=ref.center, destination=(0, 0), axis="x")

        elif justify == "right":
            ref.xmax = 0
    t.flatten()
    return t


if __name__ == "__main__":
    # c2 = text_freetype("hello", layers=[(1, 0), (2, 0)])
    # c2 = text_freetype("hello", font="Times New Roman")
    # print(c2.name)
    c2 = text_freetype()
    c2.show()

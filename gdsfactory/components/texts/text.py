from __future__ import annotations

__all__ = ["TextFont", "text", "text_klayout", "text_lines"]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import kfactory as kf
import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.constants import _glyph, _indent, _width
from gdsfactory.typings import Coordinate, LayerSpec, LayerSpecs


@dataclass(frozen=True, eq=False)
class TextFont:
    """Polygon font data for :func:`text`.

    Coordinates, widths, indents, and ``space_width`` use the same 1000-unit
    em square as the built-in DEPLOF font. ``name`` identifies the font in
    cell names and must uniquely identify its contents.
    """

    name: str
    glyphs: Mapping[str, Sequence[Sequence[Coordinate]]]
    widths: Mapping[str, float]
    indents: Mapping[str, float]
    space_width: float = 500

    def __hash__(self) -> int:
        """Hash fonts by their stable name."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Compare fonts by their stable name."""
        return isinstance(other, TextFont) and self.name == other.name


@gf.cell_with_module_name(tags=["texts"])
def text(
    text: str = "abcd",
    size: float = 10.0,
    position: Coordinate = (0, 0),
    justify: str = "left",
    layer: LayerSpec = "WG",
    font: TextFont | None = None,
) -> Component:
    """Text shapes.

    Args:
        text: string.
        size: in um of each character.
        position: x, y position.
        justify: left, right, center.
        layer: for the text.
        font: optional named polygon font. Uses the built-in DEPLOF font by default.
    """
    scaling = size / 1000
    xoffset = position[0]
    yoffset = position[1]
    t = gf.Component()

    for line in text.split("\n"):
        label = gf.Component()
        for c in line:
            ascii_val = ord(c)
            if c == " ":
                xoffset += (font.space_width if font else 500) * scaling
            elif font is not None:
                if c not in font.glyphs:
                    raise ValueError(
                        f"No glyph for character {c!r} in font {font.name!r}"
                    )
                if c not in font.widths or c not in font.indents:
                    raise ValueError(
                        f"Missing width or indent for character {c!r} "
                        f"in font {font.name!r}"
                    )
                for poly in font.glyphs[c]:
                    xpts = np.array(poly)[:, 0] * scaling
                    ypts = np.array(poly)[:, 1] * scaling
                    label.add_polygon(
                        list(zip(xpts + xoffset, ypts + yoffset, strict=False)),
                        layer=layer,
                    )
                xoffset += (font.widths[c] + font.indents[c]) * scaling
            elif 33 <= ascii_val <= 126:
                for poly in _glyph[ascii_val]:
                    xpts = np.array(poly)[:, 0] * scaling
                    ypts = np.array(poly)[:, 1] * scaling
                    label.add_polygon(
                        list(zip(xpts + xoffset, ypts + yoffset, strict=False)),
                        layer=layer,
                    )
                xoffset += (_width[ascii_val] + _indent[ascii_val]) * scaling
            else:
                raise ValueError(f"No character with ascii value {ascii_val!r}")
        t.add_ref(label)
        yoffset -= 1500 * scaling
        xoffset = position[0]
    justify = justify.lower()
    for instance in t.insts:
        if justify == "left":
            instance.xmin = position[0]
        elif justify == "right":
            instance.xmax = position[0]
        elif justify == "center":
            xmin = position[0] - instance.xsize / 2
            instance.xmin = xmin
        else:
            raise ValueError(
                f"justify = {justify!r} not in ('center', 'right', 'left')"
            )
    t.flatten()
    return t


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
    c = gf.Component()

    for i, texti in enumerate(text):
        t = gf.c.text_rectangular(text=texti, size=size, layer=layer)
        tref = c.add_ref(t)
        tref.movey(-6 * size * (i + 1))
    return c


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
    c = gf.Component()
    gen = kf.kdb.TextGenerator.default_generator()
    reg = gen.text(text, kf.kcl.dbu)

    layers = layers or [layer]

    for text_layer in layers:
        c.shapes(gf.get_layer(text_layer)).insert(reg)

    for bbox_layer in bbox_layers or []:
        c.shapes(gf.get_layer(bbox_layer)).insert(reg.bbox())
    return c

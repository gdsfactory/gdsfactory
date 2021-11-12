from typing import Tuple

import numpy as np
from phidl.geometry import _glyph, _indent, _width

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.manhattan_font import manhattan_text
from gdsfactory.name import clean_name
from gdsfactory.tech import LAYER
from gdsfactory.types import Coordinate, Layer


@gf.cell
def text(
    text: str = "abcd",
    size: float = 10.0,
    position: Coordinate = (0, 0),
    justify: str = "left",
    layer: Tuple[int, int] = LAYER.TEXT,
) -> Component:
    """Text shapes.

    Args:
        text:
        size:
        position:
        justify: left, right, center
        layer:

    """
    scaling = size / 1000
    xoffset = position[0]
    yoffset = position[1]
    t = gf.Component(f"{clean_name(text)}_{int(position[0])}_{int(position[1])}")

    for i, line in enumerate(text.split("\n")):
        label = gf.Component(f"{t.name}_{i}")
        for c in line:
            ascii_val = ord(c)
            if c == " ":
                xoffset += 500 * scaling
            elif 33 <= ascii_val <= 126:
                for poly in _glyph[ascii_val]:
                    xpts = np.array(poly)[:, 0] * scaling
                    ypts = np.array(poly)[:, 1] * scaling
                    label.add_polygon([xpts + xoffset, ypts + yoffset], layer=layer)
                xoffset += (_width[ascii_val] + _indent[ascii_val]) * scaling
            else:
                ValueError(f"[PHIDL] text(): No character with ascii value {ascii_val}")
        ref = t.add_ref(label)
        t.absorb(ref)
        yoffset -= 1500 * scaling
        xoffset = position[0]
    justify = justify.lower()
    for label in t.references:
        if justify == "left":
            pass
        elif justify == "right":
            label.xmax = position[0]
        elif justify == "center":
            label.move(origin=label.center, destination=position, axis="x")
        else:
            raise ValueError(f"justify = {justify} not in ('center', 'right', 'left')")
    return t


@gf.cell
def githash(
    text: Tuple[str, ...] = ("",),
    size: float = 0.4,
    hash_length: int = 6,
    layer: Layer = LAYER.WG,
) -> Component:
    """Returns the repo git hash
    allows a list of text, that will print on separate lines

    Args:
        text:
        size:
        hash_length:
        layer:

    """
    try:

        git_hash = gf.CONFIG["repo"][:hash_length]
        git_hash = f"gf_{git_hash}"
    except Exception:
        git_hash = f"gf_{gf.__version__}"

    c = gf.Component()
    t = manhattan_text(text=git_hash, size=size, layer=layer)
    tref = c.add_ref(t)
    c.absorb(tref)

    for i, texti in enumerate(text):
        t = manhattan_text(text=texti, size=size, layer=layer)
        tref = c.add_ref(t)
        tref.movey(-6 * size * (i + 1))
        c.absorb(tref)
    return c


if __name__ == "__main__":
    c = text(
        text=".[,ABCDEFGHIKKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:/",
        size=4.0,
        justify="right",
        position=(120.5, 3),
    )
    # c = githash(text=["a", "b"], size=10)
    c.show()

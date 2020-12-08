from typing import Tuple

import numpy as np
from phidl.geometry import _glyph, _indent, _width

import pp
from pp.component import Component
from pp.components.manhattan_font import manhattan_text
from pp.layers import LAYER
from pp.name import clean_name


def text(
    text: str = "abcd",
    size: float = 10.0,
    position: Tuple[int, int] = (0, 0),
    justify: str = "left",
    layer: Tuple[int, int] = LAYER.TEXT,
) -> Component:
    """Text shapes.

    .. plot::
      :include-source:

      import pp

      c = pp.c.text(text="abcd", size=10, position=(0, 0), justify="left", layer=1)
      pp.plotgds(c)

    """
    scaling = size / 1000
    xoffset = position[0]
    yoffset = position[1]
    t = pp.Component(
        name=clean_name(text) + "_{}_{}".format(int(position[0]), int(position[1]))
    )
    for i, line in enumerate(text.split("\n")):
        label = pp.Component(name=t.name + "{}".format(i))
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
                ValueError(
                    "[PHIDL] text(): No glyph for character with ascii value %s"
                    % ascii_val
                )
        t.add_ref(label)
        yoffset -= 1500 * scaling
        xoffset = position[0]
    justify = justify.lower()
    for label in t.references:
        if justify == "left":
            pass
        if justify == "right":
            label.xmax = position[0]
        if justify == "center":
            label.move(origin=label.center, destination=position, axis="x")
    return t


@pp.cell
def githash(text=[], size=0.4, hash_length=6, layer=LAYER.WG):
    """ returns the photonics_pdk git hash
    allows a list of text, that will print on separate lines ::

    text = [
        "sw_{}".format(Repo(CONFIG["repo"]).head.object.hexsha[:length]),
        "ap_{}".format(Repo(ap.CONFIG["repo"]).head.object.hexsha[:length]),
        "mm_{}".format(Repo(mm.CONFIG["repo"]).head.object.hexsha[:length]),
    ]
    c = githash(text=text)
    pp.write_gds(c)
    pp.show(c)

    """
    try:
        git_hash = "pp_{}".format(pp.CONFIG["repo"][:hash_length])
    except Exception:
        git_hash = "pp_{}".format(pp.__version__)

    c = pp.Component()
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
        position=(120.5, 3),
    )
    c = githash(text=["a", "b"], size=10)
    pp.show(c)

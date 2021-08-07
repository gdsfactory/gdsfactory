"""
The choice of font file is configurable from the YAML file
"""
import functools
from pathlib import Path
from typing import Dict, Tuple

import klayout.db as pya
from klayout.dbcore import Cell

import gdsfactory as gf
from gdsfactory.autoplacer.helpers import import_cell, load_gds

FONT_PATH = gf.CONFIG.get("font_path")


@functools.lru_cache()
def load_alphabet(filepath: Path = FONT_PATH) -> Dict[str, Cell]:
    c = load_gds(filepath)
    return {_c.name: _c for _c in c.each_cell()}


def add_text(
    cell: Cell,
    text: str,
    position: Tuple[int, int] = (0, 0),
    align_x: str = "center",
    align_y: str = "top",
    fontpath: Path = FONT_PATH,
) -> Cell:
    """add text label"""
    text = text.upper()
    alphabet = load_alphabet(filepath=fontpath)
    idbu = 1 / cell.layout().dbu
    si = alphabet["A"].dbbox()
    x, y = position
    w = si.width()
    h = si.height()
    c = cell.layout().create_cell("TEXT_{}".format(text))
    n = len(text)

    if align_x == "center":
        dx = -(n + 0.5) * w / 2
    elif align_x == "right":
        dx = -n * w
    else:
        dx = 0

    if align_y == "top":
        dy = -h
    elif align_y == "center":
        dy = -h / 2
    else:
        dy = 0

    for i, char in enumerate(text):
        _l = import_cell(cell.layout(), alphabet[char])
        _transform = pya.DTrans((i * w + dx) * idbu, dy * idbu)
        label = pya.CellInstArray(_l.cell_index(), _transform)
        c.insert(label)

    cell.insert(pya.CellInstArray(c.cell_index(), pya.DTrans(x, y)))
    return c


def test_alphabet() -> None:
    ly = pya.Layout()
    top_cell = ly.create_cell("TOP")
    add_text(top_cell, "HELLO-WORLD_0123456789+_-")
    # add_text(top_cell, "HELLO - WORLD 0123456789 +_-") # Need to add spaces
    # to the font
    top_cell.write("hello_world.gds")


if __name__ == "__main__":
    test_alphabet()

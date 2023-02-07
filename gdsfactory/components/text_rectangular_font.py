from __future__ import annotations

from functools import lru_cache
from typing import Dict

import numpy as np

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec

character_a = """
 XXX
X   X
XXXXX
X   X
X   X

"""


@cell
def pixel_array(
    pixels: str = character_a,
    pixel_size: float = 10.0,
    layer: LayerSpec = "M1",
) -> Component:
    """Returns a pixel component from a string representing the pixels.

    Args:
        pixels: string representing the pixels
        pixel_size: width/height for each pixel
        layer: layer for each pixel
    """
    component = Component()
    lines = [line for line in pixels.split("\n") if len(line) > 0]
    lines.reverse()
    i = 0
    i_max = 0
    a = pixel_size
    for j, line in enumerate(lines):
        i = 0
        for c in line:
            if c in ["X", "1"]:
                p0 = np.array([i * a, j * a])
                pixel = [p0 + p for p in [(0, 0), (a, 0), (a, a), (0, a)]]
                component.add_polygon(pixel, layer=layer)
            i += 1
        i_max = max(i_max, i)
    return component


FONT = """\
A
1	1	1	1	1
1	0	0	0	1
1	1	1	0	1
1	0	0	0	1
1	0	0	0	1
B
1	1	1	1	1
1	0	0	0	1
1	0	1	1	1
1	0	0	0	1
1	0	1	1	1
C
1	1	1	1	1
1	0	0	0	1
1	0	0	0	0
1	0	0	0	1
1	1	1	1	1
D
1	1	1	1	1
1	0	0	0	1
1	0	0	0	1
1	0	0	0	1
1	0	1	1	1
E
1	1	1	1	1
1	0	0	0	0
1	0	1	1	0
1	0	0	0	0
1	0	1	1	1
F
1	1	1	1	1
1	0	0	0	0
1	0	1	1	0
1	0	0	0	0
1	0	0	0	0
G
1	1	1	1	1
1	0	0	0	0
1	0	1	1	1
1	0	0	0	1
1	1	1	1	1
H
1	0	0	0	1
1	0	0	0	1
1	1	1	1	1
1	0	0	0	1
1	0	0	0	1
I
1	1	1	1	1
0	0	1	0	0
0	0	1	0	0
0	0	1	0	0
1	1	1	1	1
J
0	0	0	0	1
0	0	0	0	1
0	0	0	0	1
1	0	0	0	1
1	1	1	1	1
K
1	0	0	0	1
1	0	0	1	1
1	1	1	1	0
1	0	0	1	1
1	0	0	0	1
L
1	0	0	0	0
1	0	0	0	0
1	0	0	0	0
1	0	0	0	0
1	1	1	1	1
M
1	1	0	1	1
1	1	1	1	1
1	0	1	0	1
1	0	0	0	1
1	0	0	0	1
N
1	1	1	0	1
1	0	1	0	1
1	0	1	1	1
1	0	0	1	1
1	0	0	0	1
O
1	1	1	1	1
1	0	0	0	1
1	0	0	0	1
1	0	0	0	1
1	1	0	1	1
P
1	1	1	1	1
1	0	0	0	1
1	0	1	1	1
1	0	0	0	0
1	0	0	0	0
Q
1	1	1	1	0
1	0	0	1	0
1	0	0	1	0
1	0	0	1	0
1	1	1	1	1
R
1	1	1	1	0
1	0	0	1	0
1	0	1	1	1
1	0	0	0	1
1	0	0	0	1
S
1	1	1	1	1
1	1	0	0	0
0	1	1	1	0
0	0	0	1	1
1	1	1	1	1
T
1	1	1	1	1
0	0	1	0	0
0	0	1	0	0
0	0	1	0	0
0	0	1	0	0
U
1	0	0	0	1
1	0	0	0	1
1	0	0	0	1
1	0	0	0	1
1	1	1	1	1
V
1	0	0	0	1
0	0	0	0	0
0	1	0	1	0
0	0	0	0	0
0	0	1	0	0
W
1	0	0	0	1
1	0	0	0	1
1	0	1	0	1
1	1	1	1	1
1	1	0	1	1
X
1	1	0	1	1
1	1	1	1	1
0	1	1	1	0
1	1	1	1	1
1	1	0	1	1
Y
1	0	0	0	1
1	0	0	0	1
1	1	1	1	1
0	0	1	0	0
0	0	1	0	0
Z
1	1	1	1	1
0	0	0	1	1
0	0	1	1	0
0	1	1	0	0
1	1	1	1	1
1
0	1	1	0	0
0	0	1	0	0
0	0	1	0	0
0	0	1	0	0
0	1	1	1	0
2
1	1	1	1	1
0	0	0	1	1
1	1	1	1	1
1	1	0	0	0
1	1	1	1	1
3
1	1	1	1	1
0	0	0	1	1
1	1	1	1	1
0	0	0	1	1
1	1	1	1	1
4
1	0	0	1	1
1	0	0	1	1
1	1	1	1	1
0	0	0	1	1
0	0	0	1	1
5
1	1	1	1	1
1	1	0	0	0
1	1	1	1	1
0	0	0	1	1
1	1	1	1	1
6
1	1	1	1	1
1	0	0	0	0
1	1	1	1	1
1	1	0	1	1
1	1	1	1	1
7
1	1	1	1	1
0	0	0	1	1
0	0	0	1	1
0	0	0	1	1
0	0	0	1	1
8
1	1	1	1	1
1	1	0	1	1
1	1	1	1	1
1	1	0	1	1
1	1	1	1	1
9
1	1	1	1	1
1	1	0	1	1
1	1	1	1	1
0	0	0	1	1
0	0	0	1	1
0
0	1	1	1	1
0	1	0	0	1
0	1	0	0	1
0	1	0	0	1
0	1	1	1	1
+
0	0	0	0	0
0	0	1	0	0
0	1	1	1	0
0	0	1	0	0
0	0	0	0	0
-
0	0	0	0	0
0	0	0	0	0
0	1	1	1	0
0	0	0	0	0
0	0	0	0	0
_
0	0	0	0	0
0	0	0	0	0
0	0	0	0	0
0	0	0	0	0
0	1	1	1	0
.
0	0	0	0	0
0	0	0	0	0
0	0	0	0	0
0	0	0	0	0
0	0	1	0	0

0	0	0	0	0
0	0	0	0	0
0	0	0	0	0
0	0	0	0	0
0	0	0	0	0
"""


@lru_cache(maxsize=None)
def rectangular_font() -> Dict[str, str]:
    """Returns a rectangular font dict The keys of the dictionary are the.

    characters The values are the pixel representation of the character.
    """
    characters = {}
    lines = FONT.split("\n")
    while lines:
        line = lines.pop(0)
        if not line:
            break
        charac = line[0]

        pixels = "".join(
            lines.pop(0).replace("\t", "").replace(" ", "") + "\n" for _i in range(5)
        )

        characters[charac] = pixels
    return characters


if __name__ == "__main__":
    c = pixel_array()
    c.show(show_ports=True)

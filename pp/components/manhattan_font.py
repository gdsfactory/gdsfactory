from typing import List, Tuple

import numpy as np
from omegaconf.listconfig import ListConfig

import pp
from pp.component import Component
from pp.layers import LAYER
from pp.name import clean_name


@pp.cell
def manhattan_text(
    text: str = "abcd",
    size: float = 10.0,
    position: Tuple[float, float] = (0.0, 0.0),
    justify: str = "left",
    layer: Tuple[int, int] = LAYER.M1,
    layers_cladding: List[ListConfig] = None,
    cladding_offset: float = pp.conf.tech.cladding_offset,
) -> Component:
    """Pixel based font, guaranteed to be manhattan, without accute angles.

    .. plot::
      :include-source:

      import pp

      c = pp.c.manhattan_text(text="abcd", size=10, position=(0, 0), justify="left", layer=1)
      pp.plotgds(c)

    """
    pixel_size = size
    xoffset = position[0]
    yoffset = position[1]
    t = pp.Component(
        name=clean_name(text) + "_{}_{}".format(int(position[0]), int(position[1]))
    )
    for i, line in enumerate(text.split("\n")):
        component = pp.Component(name=t.name + "{}".format(i))
        for c in line:
            try:
                if c not in CHARAC_MAP:
                    c = c.upper()
                pixels = CHARAC_MAP[c]
            except BaseException:
                print(
                    "character {} could not be written (probably not part of dictionnary)".format(
                        c
                    )
                )
                continue

            _c = component.add_ref(
                pixel_array(pixels=pixels, pixel_size=pixel_size, layer=layer)
            )
            _c.move((xoffset, yoffset))
            component.absorb(_c)
            xoffset += pixel_size * 6

        t.add_ref(component)
        yoffset -= pixel_size * 6
        xoffset = position[0]
    justify = justify.lower()
    for ref in t.references:
        if justify == "left":
            pass
        if justify == "right":
            ref.xmax = position[0]
        if justify == "center":
            ref.move(origin=ref.center, destination=position, axis="x")

    points = [
        [t.xmin - cladding_offset / 2, t.ymin - cladding_offset],
        [t.xmax + cladding_offset / 2, t.ymin - cladding_offset],
        [t.xmax + cladding_offset / 2, t.ymax + cladding_offset],
        [t.xmin - cladding_offset / 2, t.ymax + cladding_offset],
    ]
    if layers_cladding:
        for layer in layers_cladding:
            t.add_polygon(points, layer=layer)
    return t


@pp.cell
def pixel_array(
    pixels: str = """
     XXX
    X   X
    XXXXX
    X   X
    X   X
    """,
    pixel_size: float = 10.0,
    layer: ListConfig = LAYER.M1,
) -> Component:
    component = pp.Component()
    lines = [line for line in pixels.split("\n") if len(line) > 0]
    lines.reverse()
    j = 0
    i = 0
    i_max = 0
    a = pixel_size
    for line in lines:
        i = 0
        for c in line:
            if c in ["X", "1"]:
                p0 = np.array([i * a, j * a])
                pixel = [p0 + p for p in [(0, 0), (a, 0), (a, a), (0, a)]]
                component.add_polygon(pixel, layer=layer)
            i += 1
        i_max = max(i_max, i)
        j += 1

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
1	0	0	0	1
1	0	0	0	1
0	1	0	1	0
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

CHARAC_MAP = {}


def load_font() -> None:
    lines = FONT.split("\n")
    global CHARAC_MAP
    while lines:
        line = lines.pop(0)
        if not line:
            break
        charac = line[0]

        pixels = ""
        for i in range(5):
            pixels += lines.pop(0).replace("\t", "").replace(" ", "") + "\n"

        CHARAC_MAP[charac] = pixels


load_font()


if __name__ == "__main__":
    c = manhattan_text(
        text="The mask is nearly done. only 12345 drc errors remaining",
        layers_cladding=[(33, 44)],
    )
    pp.show(c)

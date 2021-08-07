""" test critical dimension for width and space
"""
from typing import List, Optional, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.manhattan_font import manhattan_text
from gdsfactory.tech import LAYER


def square_middle(side=0.5, layer=LAYER.WG):
    component = gf.Component()
    a = side / 2
    component.add_polygon([(-a, -a), (a, -a), (a, a), (-a, a)], layer=layer)
    return component


def rectangle(width, height, layer=LAYER.WG):
    component = gf.Component()
    a = width / 2
    b = height / 2
    component.add_polygon([(-a, -b), (a, -b), (a, b), (-a, b)], layer=layer)
    return component


def triangle_middle_up(side=0.5, layer=LAYER.WG):
    component = gf.Component()
    a = side / 2
    component.add_polygon([(-a, -a), (a, -a), (0, a)], layer=layer)
    return component


def triangle_middle_down(side=0.5, layer=LAYER.WG):
    component = gf.Component()
    a = side / 2
    component.add_polygon([(-a, a), (a, a), (0, -a)], layer=layer)
    return component


@gf.cell
def char_H(
    layer: Tuple[int, int] = LAYER.WG,
    layers_cladding: Optional[List[Tuple[int, int]]] = None,
) -> Component:
    return manhattan_text(text="H", size=0.4, layer=layer)


@gf.cell
def char_L(
    layer: Tuple[int, int] = LAYER.WG,
    layers_cladding: Optional[List[Tuple[int, int]]] = None,
) -> Component:
    return manhattan_text(text="L", size=0.4, layer=layer)


CENTER_SHAPES_MAP = {
    "S": square_middle,
    "U": triangle_middle_up,
    "D": triangle_middle_down,
    "H": char_H,
    "L": char_L,
}


if __name__ == "__main__":
    # c = square_middle()
    c = char_L()
    c.show()

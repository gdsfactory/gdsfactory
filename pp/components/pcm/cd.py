""" test critical dimension for width and space
"""


import pp
from pp.layers import LAYER


def square_middle(side=0.5, layer=LAYER.WG):
    component = pp.Component()
    a = side / 2
    component.add_polygon([(-a, -a), (a, -a), (a, a), (-a, a)], layer=layer)
    return component


def rectangle(width, height, layer=LAYER.WG):
    component = pp.Component()
    a = width / 2
    b = height / 2
    component.add_polygon([(-a, -b), (a, -b), (a, b), (-a, b)], layer=layer)
    return component


def triangle_middle_up(side=0.5, layer=LAYER.WG):
    component = pp.Component()
    a = side / 2
    component.add_polygon([(-a, -a), (a, -a), (0, a)], layer=layer)
    return component


def triangle_middle_down(side=0.5, layer=LAYER.WG):
    component = pp.Component()
    a = side / 2
    component.add_polygon([(-a, a), (a, a), (0, -a)], layer=layer)
    return component


CENTER_SHAPES_MAP = {
    "S": square_middle,
    "U": triangle_middle_up,
    "D": triangle_middle_down,
}


if __name__ == "__main__":
    c = square_middle()
    pp.show(c)

from typing import Tuple, Union

from pp.cell import cell
from pp.component import Component
from pp.layers import LAYER


@cell
def L(
    width: Union[int, float] = 1,
    size: Tuple[int, int] = (10, 20),
    layer: Tuple[int, int] = LAYER.M3,
) -> Component:
    """Generates an 'L' geometry with ports on both ends. Based on phidl.

    Args:
        width: of the line
        size: length and height of the base
        layer:

    """
    D = Component()
    w = width / 2
    s1, s2 = size
    points = [(-w, -w), (s1, -w), (s1, w), (w, w), (w, s2), (-w, s2), (-w, -w)]
    D.add_polygon(points, layer=layer)
    D.add_port(name=1, midpoint=(0, s2), width=width, orientation=90)
    D.add_port(name=2, midpoint=(s1, 0), width=width, orientation=0)
    return D


if __name__ == "__main__":

    c = L()
    c.show()

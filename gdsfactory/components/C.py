from typing import Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.tech import LAYER


@gf.cell
def C(
    width: float = 1.0,
    size: Tuple[float, float] = (10.0, 20.0),
    layer: Tuple[int, int] = LAYER.M3,
) -> Component:
    """Generates a 'C' geometry with ports on both ends. Adapted from phidl

    Args:
        width: of the line
        size: length and height of the base
        layer:

    """
    D = Component()
    w = width / 2
    s1, s2 = size
    points = [
        (-w, -w),
        (s1, -w),
        (s1, w),
        (w, w),
        (w, s2 - w),
        (s1, s2 - w),
        (s1, s2 + w),
        (-w, s2 + w),
        (-w, -w),
    ]
    D.add_polygon(points, layer=layer)
    D.add_port(name="o1", midpoint=(s1, s2), width=width, orientation=0)
    D.add_port(name="o2", midpoint=(s1, 0), width=width, orientation=0)
    return D


if __name__ == "__main__":

    c = C(width=1.0)
    c.show()

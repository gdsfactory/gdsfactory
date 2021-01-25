from typing import List, Tuple

from pp.cell import cell
from pp.component import Component


@cell
def bbox(
    bbox: List[Tuple[int, int]] = [(-1, -1), (3, 4)], layer: Tuple[int, int] = (1, 0)
) -> Component:
    """ Creates a bounding box rectangle from coordinates, to allow
    creation of a rectangle bounding box directly form another shape.

    Args:
        bbox: Coordinates of the box [(x1, y1), (x2, y2)].
        layer:

    """
    D = Component(name="bbox")
    (a, b), (c, d) = bbox
    points = ((a, b), (c, b), (c, d), (a, d))
    D.add_polygon(points, layer=layer)
    return D


if __name__ == "__main__":
    import pp

    c = pp.c.L()
    c << bbox(bbox=c.bbox)
    pp.show(c)

from typing import Tuple, Union

from numpy import array

from pp.cell import cell
from pp.component import Component


@cell
def bbox(
    bbox: Union[array, Tuple[Tuple[float]]] = ((-1.0, -1.0), (3.0, 4.0)),
    layer: Tuple[int, int] = (1, 0),
) -> Component:
    """Returns bounding box rectangle from coordinates, to allow
    creation of a rectangle bounding box directly from another shape.

    Args:
        bbox: Coordinates of the box [(x1, y1), (x2, y2)].
        layer:

    """
    D = Component()
    (a, b), (c, d) = bbox
    points = ((a, b), (c, b), (c, d), (a, d))
    D.add_polygon(points, layer=layer)
    return D


if __name__ == "__main__":
    import pp

    c = pp.components.L()
    c << bbox(bbox=c.bbox)
    # c = bbox()
    c.show()

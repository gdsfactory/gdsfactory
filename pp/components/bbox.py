from typing import Tuple, Union

from numpy import array

import pp

Coordinate = Union[Tuple[float, float], array]


@pp.cell_without_validator
def bbox(
    bbox: Tuple[Coordinate, Coordinate] = ((-1.0, -1.0), (3.0, 4.0)),
    layer: Tuple[int, int] = (1, 0),
) -> pp.Component:
    """Returns bounding box rectangle from coordinates, to allow
    creation of a rectangle bounding box directly from another shape.

    Args:
        bbox: Coordinates of the box [(x1, y1), (x2, y2)].
        layer:

    """
    D = pp.Component()
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

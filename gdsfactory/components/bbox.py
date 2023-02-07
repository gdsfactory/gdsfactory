from __future__ import annotations

from typing import Tuple, Union

from numpy import array

import gdsfactory as gf

Float2 = Tuple[float, float]
Coordinate = Union[Tuple[Float2, Float2], array]


@gf.cell_without_validator
def bbox(
    bbox: Tuple[Coordinate, Coordinate] = ((-1.0, -1.0), (3.0, 4.0)),
    layer: Tuple[int, int] = (1, 0),
    top: float = 0,
    bottom: float = 0,
    left: float = 0,
    right: float = 0,
) -> gf.Component:
    """Returns bounding box rectangle from coordinates.

    Args:
        bbox: Coordinates of the box [(x1, y1), (x2, y2)].
        layer: for bbox.
        top: north offset.
        bottom: south offset.
        left: west offset.
        right: east offset.
    """
    D = gf.Component()
    (xmin, ymin), (xmax, ymax) = bbox
    points = [
        [xmin - left, ymin - bottom],
        [xmax + right, ymin - bottom],
        [xmax + right, ymax + top],
        [xmin - left, ymax + top],
    ]
    D.add_polygon(points, layer=layer)
    return D


if __name__ == "__main__":
    c = gf.Component()
    a = c << gf.components.L()
    c << bbox(bbox=a.bbox, top=10, left=5, right=-2)
    c.show(show_ports=True)

from __future__ import annotations

import gdsfactory as gf

Float2 = tuple[float, float]
Coordinate = tuple[Float2, Float2]


def bbox_to_points(
    bbox,
    top: float = 0,
    bottom: float = 0,
    left: float = 0,
    right: float = 0,
) -> list[list[float]]:
    (xmin, ymin), (xmax, ymax) = bbox
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    return [
        [xmin - left, ymin - bottom],
        [xmax + right, ymin - bottom],
        [xmax + right, ymax + top],
        [xmin - left, ymax + top],
    ]


@gf.cell_without_validator
def bbox(
    bbox: tuple[Coordinate, Coordinate] = ((-1.0, -1.0), (3.0, 4.0)),
    layer: tuple[int, int] = (1, 0),
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
    from gdsfactory.generic_tech import get_generic_pdk

    PDK = get_generic_pdk()
    PDK.activate()
    c = gf.Component()
    a = c << gf.components.L()
    c << bbox(bbox=a.bbox, top=10, left=5, right=-2)
    c.show(show_ports=True)

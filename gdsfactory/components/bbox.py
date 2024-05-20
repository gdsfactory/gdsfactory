from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import LayerSpec


def bbox_to_points(
    bbox,
    top: float = 0,
    bottom: float = 0,
    left: float = 0,
    right: float = 0,
) -> list[list[float]]:
    """Returns bounding box rectangle with offsets."""
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


@gf.cell
def bbox(
    component: gf.Component | gf.Instance = "L",
    layer: LayerSpec = (1, 0),
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
    c = gf.Component()
    bbox = component.dbbox()
    xmin, ymin, xmax, ymax = bbox.left, bbox.bottom, bbox.right, bbox.top
    points = [
        [xmin - left, ymin - bottom],
        [xmax + right, ymin - bottom],
        [xmax + right, ymax + top],
        [xmin - left, ymax + top],
    ]
    c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    # c = gf.Component()
    # a1 = c << gf.components.L()
    # _ = c << bbox(a1, top=10, left=5, right=-2)
    c = bbox()
    c.show()

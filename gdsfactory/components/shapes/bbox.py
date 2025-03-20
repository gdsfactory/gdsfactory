from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import ComponentReference
from gdsfactory.typings import LayerSpec


def bbox_to_points(
    bbox: gf.kdb.DBox,
    top: float = 0,
    bottom: float = 0,
    left: float = 0,
    right: float = 0,
) -> list[tuple[float, float]]:
    """Returns bounding box rectangle with offsets.

    Args:
        bbox: DBbox.
        top: north offset.
        bottom: south offset.
        left: west offset.
        right: east offset.

    """
    xmin, ymin, xmax, ymax = bbox.left, bbox.bottom, bbox.right, bbox.top
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    return [
        (xmin - left, ymin - bottom),
        (xmax + right, ymin - bottom),
        (xmax + right, ymax + top),
        (xmin - left, ymax + top),
    ]


@gf.cell
def bbox(
    component: gf.Component | ComponentReference,
    layer: LayerSpec,
    top: float = 0,
    bottom: float = 0,
    left: float = 0,
    right: float = 0,
) -> gf.Component:
    """Returns bounding box rectangle from coordinates.

    Args:
        component: component or instance to get bbox from.
        layer: for bbox.
        top: north offset.
        bottom: south offset.
        left: west offset.
        right: east offset.
    """
    c = gf.Component()
    if not isinstance(component, ComponentReference):
        component = gf.get_component(component)

    bbox = component.dbbox()
    xmin, ymin, xmax, ymax = bbox.left, bbox.bottom, bbox.right, bbox.top
    points = [
        (xmin - left, ymin - bottom),
        (xmax + right, ymin - bottom),
        (xmax + right, ymax + top),
        (xmin - left, ymax + top),
    ]
    c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    # c = gf.Component()
    # l= gf.components.L()
    r = gf.c.text()
    b = bbox(r, layer=(2, 0))
    c = gf.Component()
    c << r
    c << b
    c.show()

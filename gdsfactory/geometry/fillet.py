from __future__ import annotations

from typing import Union, List

import gdstk

from gdsfactory.typings import Component, ComponentReference


def fillet(
    operand: Union[ComponentReference, Component, gdstk.Polygon, List[gdstk.Polygon]],
    radius: Union[float, List[float]],
    tolerance: float = 0.01,
) -> List[gdstk.Polygon]:
    """Perform a fillet operation and return the list of resulting Polygons.

    Args:
        operand: polygon, list of Polygons, Component, or ComponentReference.
        radius: Fillet radius. You can also define a value for each vertex.
        tolerance: for calculating the polygonal approximation of the filleted corners.

    """

    if hasattr(operand, "get_polygons"):
        operand = operand.get_polygons(as_array=False)

    elif isinstance(operand, gdstk.Polygon):
        operand = [operand]

    return [polygon.fillet(radius=radius, tolerance=tolerance) for polygon in operand]


if __name__ == "__main__":
    import gdsfactory as gf

    points = [(0, 0), (1.2, 0), (1.2, 0.3), (1, 0.3), (1.5, 1), (0, 1.5)]
    o = gdstk.Polygon(points, datatype=1)

    o = gf.components.mzi()
    p = fillet(o, radius=0.3)
    c = gf.Component("demo")
    c.add_polygon(p)
    c.show(show_ports=True)

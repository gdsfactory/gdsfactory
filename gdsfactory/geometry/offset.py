from typing import Tuple

import gdspy
import numpy as np
from phidl.geometry import (
    Device,
    DeviceReference,
    Polygon,
    _merge_floating_point_errors,
    _offset_polygons_parallel,
    _parse_layer,
)

import gdsfactory as gf
from gdsfactory.types import Component, LayerSpec


@gf.cell
def offset(
    elements: Component,
    distance: float = 0.1,
    join_first: bool = True,
    precision: float = 1e-4,
    num_divisions: Tuple[int, int] = (1, 1),
    join: str = "miter",
    tolerance: int = 2,
    max_points: int = 4000,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns an element containing all polygons with an offset Shrinks or \
    expands a polygon or set of polygons.

    adapted from phidl.geometry

    Args:
        elements: Component(/Reference), list of Component(/Reference), or Polygon
          Polygons to offset or Component containing polygons to offset.
        distance: Distance to offset polygons. Positive values expand, negative shrink.
        precision: Desired precision for rounding vertex coordinates.
        num_divisions: The number of divisions with which the geometry is divided into
          multiple rectangular regions. This allows for each region to be
          processed sequentially, which is more computationally efficient.
        join: {'miter', 'bevel', 'round'} Type of join used to create polygon offset
        tolerance: For miter joints, this number must be at least 2 represents the
          maximal distance in multiples of offset between new vertices and their
          original position before beveling to avoid spikes at acute joints. For
          round joints, it indicates the curvature resolution in number of
          points per full circle.
        max_points: The maximum number of vertices within the resulting polygon.
        layer: Specific layer to put polygon geometry on.

    Returns
        Component containing a polygon(s) with the specified offset applied.

    """
    if not isinstance(elements, list):
        elements = [elements]
    polygons_to_offset = []
    for e in elements:
        if isinstance(e, (Device, DeviceReference)):
            polygons_to_offset += e.get_polygons(by_spec=False)
        elif isinstance(e, (Polygon, gdspy.Polygon)):
            polygons_to_offset.append(e)
    if len(polygons_to_offset) == 0:
        return gf.Component("offset")
    polygons_to_offset = _merge_floating_point_errors(
        polygons_to_offset, tol=precision / 1000
    )

    layer = gf.get_layer(layer)
    gds_layer, gds_datatype = _parse_layer(layer)
    if all(np.array(num_divisions) == np.array([1, 1])):
        p = gdspy.offset(
            polygons_to_offset,
            distance=distance,
            join=join,
            tolerance=tolerance,
            precision=precision,
            join_first=join_first,
            max_points=max_points,
            layer=gds_layer,
            datatype=gds_datatype,
        )
    else:
        p = _offset_polygons_parallel(
            polygons_to_offset,
            distance=distance,
            num_divisions=num_divisions,
            join_first=join_first,
            precision=precision,
            join=join,
            tolerance=tolerance,
        )

    component = gf.Component("offset")
    polygons = component.add_polygon(p, layer=layer)
    [
        polygon.fracture(max_points=max_points, precision=precision)
        for polygon in polygons
    ]
    return component


def test_offset() -> None:
    c = gf.components.ring()
    co = offset(c, distance=0.5)
    assert int(co.area()) == 94


if __name__ == "__main__":
    c = gf.components.ring()
    co = offset(c, distance=0.5)
    gf.show(co)

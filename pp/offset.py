import gdspy
import numpy as np

from phidl.geometry import (
    Device,
    DeviceReference,
    _merge_floating_point_errors,
    _parse_layer,
    Polygon,
    _offset_polygons_parallel,
)
import pp


def offset(
    elements,
    distance=0.1,
    join_first=True,
    precision=1e-4,
    num_divisions=[1, 1],
    join="miter",
    tolerance=2,
    max_points=4000,
    layer=0,
):
    """ returns an element containing all polygons with an offset
    from phidl geometry
    """
    if type(elements) is not list:
        elements = [elements]
    polygons_to_offset = []
    for e in elements:
        if isinstance(e, (Device, DeviceReference)):
            polygons_to_offset += e.get_polygons(by_spec=False)
        elif isinstance(e, (Polygon, gdspy.Polygon)):
            polygons_to_offset.append(e)
    if len(polygons_to_offset) == 0:
        return pp.Component("offset")
    polygons_to_offset = _merge_floating_point_errors(
        polygons_to_offset, tol=precision / 1000
    )
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

    D = pp.Component("offset")
    polygons = D.add_polygon(p, layer=layer)
    [
        polygon.fracture(max_points=max_points, precision=precision)
        for polygon in polygons
    ]
    return D


if __name__ == "__main__":
    c = pp.c.ring()
    co = offset(c)
    pp.show(co)

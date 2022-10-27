"""Based on phidl.geometry."""
from typing import Tuple

import gdspy
import numpy as np
from gdspy import clipper

import gdsfactory as gf
from gdsfactory.component_layout import Polygon, _parse_layer
from gdsfactory.types import Component, ComponentReference, LayerSpec


def _merge_nearby_floating_points(x, tol=1e-10):
    """Takes an array `x` and merges any values within the tolerance `tol`.

    Args:
        x: list of int or float
            Array of values with floating point errors.
        tol : float
            Tolerance within which points will be merged.

    Returns
        xsort : list of int or float Corrected and sorted array.

    Examples
    --------
    If given:
    >>> x = [-2, -1, 0, 1.0001, 1.0002, 1.0003, 4, 5, 5.003, 6, 7, 8]
    >>> _merge_nearby_floating_points(x, tol = 1e-3)
    Will return:
    >>> [-2, -1, 0, 1.0001, 1.0001, 1.0001, 4, 5, 5.003, 6, 7, 8].
    """
    xargsort = np.argsort(x)
    xargunsort = np.argsort(xargsort)
    xsort = x[xargsort]
    xsortthreshold = np.diff(xsort) < tol
    xsortthresholdind = np.argwhere(xsortthreshold)

    # Merge nearby floating point values
    for xi in xsortthresholdind:
        xsort[xi + 1] = xsort[xi]
    return xsort[xargunsort]


def _merge_floating_point_errors(polygons, tol=1e-10):
    """Fixes floating point errors in the input polygons by merging values \
    within `tol` tolerance. See _merge_nearby_floating_points for specifics.

    Args:
        polygons: PolygonSet or list of polygons
            Set of polygons with floating point errors.
        tol: float
            Tolerance within which points will be merged.

    Returns
        polygons_fixed : PolygonSet Set of corrected polygons.
    """
    stacked_polygons = np.vstack(polygons)
    x = stacked_polygons[:, 0]
    y = stacked_polygons[:, 1]
    polygon_indices = np.cumsum([len(p) for p in polygons])

    xfixed = _merge_nearby_floating_points(x, tol=tol)
    yfixed = _merge_nearby_floating_points(y, tol=tol)
    stacked_polygons_fixed = np.vstack([xfixed, yfixed]).T
    return np.vsplit(stacked_polygons_fixed, polygon_indices[:-1])


def _crop_region(polygons, left, bottom, right, top, precision):
    """Given a rectangular boundary defined by left/bottom/right/top, this \
    takes a list of polygons and cuts them at the boundary, discarding parts \
    of the polygons outside the rectangle.

    Args:
        polygons : PolygonSet or list of polygons
            Set or list of polygons to be cropped.
        left : int or float
            The x-coordinate of the lefthand boundary.
        bottom : int or float
            The y-coordinate of the bottom boundary.
        right : int or float
            The x-coordinate of the righthand boundary.
        top : int or float
            The y-coordinate of the top boundary.
        precision : float
            Desired precision for rounding vertex coordinates.

    Returns:
        cropped_polygons : PolygonSet or list of polygons
            Set or list of polygons that are cropped according to the specified
            boundary.
    """
    cropped_polygons = []
    for p in polygons:
        clipped_polys = clipper._chop(p, [top, bottom], 1, 1 / precision)
        # polygon, [cuts], axis, scale
        for cp in clipped_polys[1]:
            result = clipper._chop(cp, [left, right], 0, 1 / precision)
            cropped_polygons += list(result[1])
    return cropped_polygons


def _crop_edge_polygons(all_polygons, bboxes, left, bottom, right, top, precision):
    """Parses out which polygons are along the edge of the rectangle and need \
    to be cropped and which are deep inside the rectangle region and can be \
    left alone, then crops only those polygons along the edge.

    Args:
        all_polygons : PolygonSet or list of polygons
            Set or list of polygons to be cropped.
        bboxes : list
            List of all polygon bboxes in all_polygons.
        left : int or float
            The x-coordinate of the lefthand boundary.
        bottom : int or float
            The y-coordinate of the bottom boundary.
        right : int or float
            The x-coordinate of the righthand boundary.
        top : int or float
            The y-coordinate of the top boundary.
        precision : float
            Desired precision for rounding vertex coordinates.

    Returns:
        polygons_to_process : PolygonSet or list of polygons
            Set or list of polygons with crop applied to edge polygons.
    """
    polygons_in_rect_i = _find_bboxes_in_rect(bboxes, left, bottom, right, top)
    polygons_edge_i = _find_bboxes_on_rect_edge(bboxes, left, bottom, right, top)
    polygons_in_rect_no_edge_i = polygons_in_rect_i & (~polygons_edge_i)

    # Crop polygons along the edge and recombine them with polygons inside the
    # rectangle
    polygons_edge = all_polygons[polygons_edge_i]
    polygons_in_rect_no_edge = all_polygons[polygons_in_rect_no_edge_i].tolist()
    polygons_edge_cropped = _crop_region(
        polygons_edge, left, bottom, right, top, precision=precision
    )
    return polygons_in_rect_no_edge + polygons_edge_cropped


def _find_bboxes_in_rect(bboxes, left, bottom, right, top):
    """Given a list of polygon bounding boxes and a rectangle defined by \
    left/bottom/right/top, this function returns those polygons which overlap \
    the rectangle.

    Args:
        bboxes : list
            List of all polygon bboxes.
        left : int or float
            The x-coordinate of the lefthand boundary.
        bottom : int or float
            The y-coordinate of the bottom boundary.
        right : int or float
            The x-coordinate of the righthand boundary.
        top : int or float
            The y-coordinate of the top boundary.

    Returns:
        result : list
            List of all polygon bboxes that overlap with the defined rectangle.
    """
    return (
        (bboxes[:, 0] <= right)
        & (bboxes[:, 2] >= left)
        & (bboxes[:, 1] <= top)
        & (bboxes[:, 3] >= bottom)
    )


# _find_bboxes_on_rect_edge
def _find_bboxes_on_rect_edge(bboxes, left, bottom, right, top):
    """Given a list of polygon bounding boxes and a rectangular boundary \
    defined by left/bottom/right/top, this function returns those polygons \
    which intersect the rectangular boundary.

    Args:
        bboxes : list
            List of all polygon bboxes.
        left : int or float
            The x-coordinate of the lefthand boundary.
        bottom : int or float
            The y-coordinate of the bottom boundary.
        right : int or float
            The x-coordinate of the righthand boundary.
        top : int or float
            The y-coordinate of the top boundary.

    Returns:
        result: List of all polygon bboxes that intersect the defined
            rectangular boundary.
    """
    bboxes_left = _find_bboxes_in_rect(bboxes, left, bottom, left, top)
    bboxes_right = _find_bboxes_in_rect(bboxes, right, bottom, right, top)
    bboxes_top = _find_bboxes_in_rect(bboxes, left, top, right, top)
    bboxes_bottom = _find_bboxes_in_rect(bboxes, left, bottom, right, bottom)
    return bboxes_left | bboxes_right | bboxes_top | bboxes_bottom


def _offset_region(
    all_polygons,
    bboxes,
    left,
    bottom,
    right,
    top,
    distance=5,
    join_first=True,
    precision=1e-4,
    join="miter",
    tolerance=2,
):
    """Taking a region of e.g. size (x, y) which needs to be offset by \
    distance d, this function crops out a region (x+2*d, y+2*d) large, offsets \
    that region, then crops it back to size (x, y) to create a valid result.

    Args:
        all_polygons : PolygonSet or list of polygons
            Set or list of polygons to be cropped and offset.
        bboxes : list
            List of all polygon bboxes in all_polygons.
        left : int or float
            The x-coordinate of the lefthand boundary.
        bottom : int or float
            The y-coordinate of the bottom boundary.
        right : int or float
            The x-coordinate of the righthand boundary.
        top : int or float
            The y-coordinate of the top boundary.
        distance : int or float
            Distance to offset polygons. Positive values expand, negative shrink.
        join_first : bool
            Join all paths before offsetting to avoid unnecessary joins in
            adjacent polygon sides.
        precision : float
            Desired precision for rounding vertex coordinates.
        join : {'miter', 'bevel', 'round'}
            Type of join used to create the offset polygon.
        tolerance : int or float
            For miter joints, this number must be at least 2 and it represents the
            maximal distance in multiples of offset between new vertices and their
            original position before beveling to avoid spikes at acute joints. For
            round joints, it indicates the curvature resolution in number of
            points per full circle.

    Returns:
        polygons_offset_cropped:
            The resulting input polygons that are cropped to be between the
            coordinates (left, bottom, right, top)

    """
    # Mark out a region slightly larger than the final desired region
    d = distance * 1.01

    polygons_to_offset = _crop_edge_polygons(
        all_polygons,
        bboxes,
        left - d,
        bottom - d,
        right + d,
        top + d,
        precision=precision,
    )

    # Offset the resulting cropped polygons and recrop to final desired size
    polygons_offset = clipper.offset(
        polygons_to_offset, distance, join, tolerance, 1 / precision, int(join_first)
    )
    return _crop_region(polygons_offset, left, bottom, right, top, precision=precision)


def _polygons_to_bboxes(polygons):
    """Generates the bboxes of all input polygons.

    Args:
        polygons : PolygonSet or list of polygons
            Set or list of polygons to generate bboxes of.

    Returns:
        bboxes : list
            List of all polygon bboxes in polygons.
    """
    #    Build bounding boxes
    bboxes = np.empty([len(polygons), 4])
    for n, p in enumerate(polygons):
        left, bottom = np.min(p, axis=0)
        right, top = np.max(p, axis=0)
        bboxes[n] = [left, bottom, right, top]
    return bboxes


def _offset_polygons_parallel(
    polygons,
    distance=5,
    num_divisions=(10, 10),
    join_first=True,
    precision=1e-4,
    join="miter",
    tolerance=2,
):
    """Offsets a list of subsections and returns the offset polygons.

    Args:
        polygons : PolygonSet or list of polygons
        distance : int or float
            Distance to offset polygons. Positive values expand, negative shrink.
        num_divisions : array-like[2] of int
            The number of divisions with which the geometry is divided into
            multiple rectangular regions. This allows for each region to be
            processed sequentially, which is more computationally efficient.
        join_first : bool
            Join all paths before offsetting to avoid unnecessary joins in
            adjacent polygon sides.
        precision : float
            Desired precision for rounding vertex coordinates.
        join : {'miter', 'bevel', 'round'}
            Type of join used to create the offset polygon.
        tolerance : int or float
            For miter joints, this number must be at least 2 and it represents the
            maximal distance in multiples of offset between new vertices and their
            original position before beveling to avoid spikes at acute joints. For
            round joints, it indicates the curvature resolution in number of
            points per full circle.

    """
    # Build bounding boxes
    polygons = np.asarray(polygons)
    bboxes = _polygons_to_bboxes(polygons)

    xmin, ymin = np.min(bboxes[:, 0:2], axis=0) - distance
    xmax, ymax = np.max(bboxes[:, 2:4], axis=0) + distance

    xsize = xmax - xmin
    ysize = ymax - ymin
    xdelta = xsize / num_divisions[0]
    ydelta = ysize / num_divisions[1]
    xcorners = xmin + np.arange(num_divisions[0]) * xdelta
    ycorners = ymin + np.arange(num_divisions[1]) * ydelta

    offset_polygons = []
    for xc in xcorners:
        for yc in ycorners:
            left = xc
            right = xc + xdelta
            bottom = yc
            top = yc + ydelta
            _offset_region_polygons = _offset_region(
                polygons,
                bboxes,
                left,
                bottom,
                right,
                top,
                distance=distance,
                join_first=join_first,
                precision=precision,
                join=join,
                tolerance=tolerance,
            )
            offset_polygons += _offset_region_polygons

    return offset_polygons


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

    Returns:
        Component containing a polygon(s) with the specified offset applied.

    """
    if not isinstance(elements, list):
        elements = [elements]
    polygons_to_offset = []
    for e in elements:
        if isinstance(e, (Component, ComponentReference)):
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

    component = gf.Component()
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
    co.show()

from typing import Tuple, Union

import gdspy
import numpy as np
from gdspy import clipper

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.component_layout import Polygon, _parse_layer
from gdsfactory.component_reference import ComponentReference
from gdsfactory.geometry.offset import _crop_edge_polygons, _polygons_to_bboxes
from gdsfactory.types import ComponentOrReference, Int2, LayerSpec


def _boolean_region(
    all_polygons_A,
    all_polygons_B,
    bboxes_A,
    bboxes_B,
    left,
    bottom,
    right,
    top,
    operation="and",
    precision: float = 1e-4,
):
    """Returns boolean for a region.

    Taking a region of e.g. size (x, y) which needs to be booleaned,
    this function crops out a region (x, y) large from each set of polygons
    (A and B), booleans that cropped region and returns the result.

    Args:
        all_polygons_A : PolygonSet or list of polygons
            Set or list of polygons to be booleaned.
        all_polygons_B : PolygonSet or list of polygons
            Set or list of polygons to be booleaned.
        bboxes_A : list
            List of all polygon bboxes in all_polygons_A
        bboxes_B : list
            List of all polygon bboxes in all_polygons_B
        left : int or float
            The x-coordinate of the lefthand boundary.
        bottom : int or float
            The y-coordinate of the bottom boundary.
        right : int or float
            The x-coordinate of the righthand boundary.
        top : int or float
            The y-coordinate of the top boundary.
        operation : {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'}
            Boolean operation to perform.
        precision : float
            Desired precision for rounding vertex coordinates.

    Returns:
        polygons_boolean : PolygonSet or list of polygons
        Set or list of polygons with boolean operation applied.
    """
    polygons_to_boolean_A = _crop_edge_polygons(
        all_polygons_A, bboxes_A, left, bottom, right, top, precision
    )
    polygons_to_boolean_B = _crop_edge_polygons(
        all_polygons_B, bboxes_B, left, bottom, right, top, precision
    )
    return clipper.clip(
        polygons_to_boolean_A, polygons_to_boolean_B, operation, 1 / precision
    )


def _boolean_polygons_parallel(
    polygons_A, polygons_B, num_divisions=(10, 10), operation="and", precision=1e-4
):
    """Returns boolean on a list of subsections of the original geometry.

    Returns list of polygons, with all the booleaned polygons from each subsection.

    Args:
        polygons_A : PolygonSet or list of polygons
            Set or list of polygons to be booleaned.
        polygons_B : PolygonSet or list of polygons
            Set or list of polygons to be booleaned.
        num_divisions : array-like[2] of int
            The number of divisions with which the geometry is divided into
            multiple rectangular regions. This allows for each region to be
            processed sequentially, which is more computationally efficient.
        operation : {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'}
            Boolean operation to perform.
        precision : float
            Desired precision for rounding vertex coordinates.

    """
    # Build bounding boxes
    polygons_A = np.asarray(polygons_A)
    polygons_B = np.asarray(polygons_B)
    bboxes_A = _polygons_to_bboxes(polygons_A)
    bboxes_B = _polygons_to_bboxes(polygons_B)

    xmin, ymin = np.min(
        [np.min(bboxes_A[:, 0:2], axis=0), np.min(bboxes_B[:, 0:2], axis=0)], axis=0
    )
    xmax, ymax = np.max(
        [np.max(bboxes_A[:, 2:4], axis=0), np.max(bboxes_B[:, 2:4], axis=0)], axis=0
    )

    xsize = xmax - xmin
    ysize = ymax - ymin
    xdelta = xsize / num_divisions[0]
    ydelta = ysize / num_divisions[1]
    xcorners = xmin + np.arange(num_divisions[0]) * xdelta
    ycorners = ymin + np.arange(num_divisions[1]) * ydelta

    boolean_polygons = []
    for xc in xcorners:
        for yc in ycorners:
            left = xc
            right = xc + xdelta
            bottom = yc
            top = yc + ydelta
            _boolean_region_polygons = _boolean_region(
                polygons_A,
                polygons_B,
                bboxes_A,
                bboxes_B,
                left,
                bottom,
                right,
                top,
                operation=operation,
                precision=precision,
            )
            boolean_polygons += _boolean_region_polygons

    return boolean_polygons


@gf.cell
def boolean(
    A: Union[ComponentOrReference, Tuple[ComponentOrReference, ...]],
    B: Union[ComponentOrReference, Tuple[ComponentOrReference, ...]],
    operation: str,
    precision: float = 1e-4,
    num_divisions: Union[int, Int2] = (1, 1),
    max_points: int = 4000,
    layer: LayerSpec = (1, 0),
) -> Component:
    """Performs boolean operations between 2 Component/Reference/list objects.

    ``operation`` should be one of {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'}.
    Note that 'A+B' is equivalent to 'or', 'A-B' is equivalent to 'not', and
    'B-A' is equivalent to 'not' with the operands switched

    based on phidl.geometry.boolean
    You can also use gdsfactory.drc.boolean_klayout

    Args:
        A: Component(/Reference) or list of Component(/References).
        B: Component(/Reference) or list of Component(/References).
        operation: {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'}.
        precision: float Desired precision for rounding vertex coordinates..
        num_divisions: number of divisions with which the geometry is divided into
          multiple rectangular regions. This allows for each region to be
          processed sequentially, which is more computationally efficient.
        max_points: The maximum number of vertices within the resulting polygon.
        layer: Specific layer to put polygon geometry on.

    Returns: Component with polygon(s) of the boolean operations between
      the 2 input Components performed.

    Notes
    -----
    'A+B' is equivalent to 'or'.
    'A-B' is equivalent to 'not'.
    'B-A' is equivalent to 'not' with the operands switched.

    """
    D = Component()
    A_polys = []
    B_polys = []
    A = list(A) if isinstance(A, (list, tuple)) else [A]
    B = list(B) if isinstance(B, (list, tuple)) else [B]

    for X, polys in ((A, A_polys), (B, B_polys)):
        for e in X:
            if isinstance(e, (Component, ComponentReference)):
                polys.extend(e.get_polygons())
            elif isinstance(e, Polygon):
                polys.extend(e.polygons)

    layer = gf.pdk.get_layer(layer)
    gds_layer, gds_datatype = _parse_layer(layer)

    operation = operation.lower().replace(" ", "")
    if operation == "a-b":
        operation = "not"
    elif operation == "b-a":
        operation = "not"
        A_polys, B_polys = B_polys, A_polys
    elif operation == "a+b":
        operation = "or"
    elif operation not in ["not", "and", "or", "xor", "a-b", "b-a", "a+b"]:
        raise ValueError(
            "gdsfactory.geometry.boolean() `operation` "
            "parameter not recognized, must be one of the "
            "following:  'not', 'and', 'or', 'xor', 'A-B', "
            "'B-A', 'A+B'"
        )

    # Check for trivial solutions
    if (not A_polys or not B_polys) and operation != "or":
        if (
            operation != "not"
            and operation != "and"
            and operation == "xor"
            and not A_polys
            and not B_polys
            or operation != "not"
            and operation == "and"
        ):
            p = None
        elif operation != "not" and operation == "xor" and not A_polys:
            p = B_polys
        elif operation != "not" and operation == "xor":
            p = A_polys
        elif operation == "not":
            p = A_polys or None
    elif not A_polys and not B_polys:
        p = None
    elif all(np.array(num_divisions) == np.array([1, 1])):
        p = gdspy.boolean(
            operand1=A_polys,
            operand2=B_polys,
            operation=operation,
            precision=precision,
            max_points=max_points,
            layer=gds_layer,
            datatype=gds_datatype,
        )
    else:
        p = _boolean_polygons_parallel(
            polygons_A=A_polys,
            polygons_B=B_polys,
            num_divisions=num_divisions,
            operation=operation,
            precision=precision,
        )

    if p is not None:
        polygons = D.add_polygon(p, layer=layer)
        [
            polygon.fracture(max_points=max_points, precision=precision)
            for polygon in polygons
        ]
    return D


def test_boolean() -> None:
    c = gf.Component()
    e1 = c << gf.components.ellipse()
    e2 = c << gf.components.ellipse(radii=(10, 6))
    e3 = c << gf.components.ellipse(radii=(10, 4))
    e3.movex(5)
    e2.movex(2)
    c = boolean(A=[e1, e3], B=e2, operation="A-B")
    assert len(c.polygons) == 2, len(c.polygons)


if __name__ == "__main__":
    c = gf.Component()
    e1 = c << gf.components.ellipse()
    e2 = c << gf.components.ellipse(radii=(10, 6))
    e3 = c << gf.components.ellipse(radii=(10, 4))
    e3.movex(5)
    e2.movex(2)
    c = boolean(A=[e1, e3], B=e2, operation="A-B")
    c.show(show_ports=True)

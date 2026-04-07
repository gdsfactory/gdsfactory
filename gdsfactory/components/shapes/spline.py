from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec
from gdsfactory.utils import spline_points


@gf.cell_with_module_name
def spline(
    points: tuple[tuple[float, float], ...] = ((0.0, 0.0), (10.0, 5.0), (20.0, 0.0)),
    npoints: int = 100,
    degree: int = 3,
    bc_type: str | None = None,
    monotonic: bool = False,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a Component with a spline-interpolated polygon.

    Args:
        points: control points.
        npoints: number of points to generate for the polygon.
        degree: spline degree.
        bc_type: boundary conditions for the spline (e.g., 'clamped', 'natural').
        monotonic: if True, uses PCHIP interpolation.
        layer: layer spec.

    .. code::

        import gdsfactory as gf
        c = gf.components.spline(points=((0, 0), (10, 5), (20, 0), (10, -5), (0, 0)))
        c.show()
    """
    method = "pchip" if monotonic else "bspline"
    pts = spline_points(
        points, degree=degree, npoints=npoints, bc_type=bc_type, method=method
    )

    c = gf.Component()
    c.add_polygon(pts, layer=layer)
    return c

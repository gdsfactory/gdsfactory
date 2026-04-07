from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeGuard

import kfactory as kf
import klayout.db as kdb
import numpy as np

if TYPE_CHECKING:
    from gdsfactory.typings import ComponentSpec, Coordinate


def to_kdb_dpoints(
    points: Sequence[Coordinate | kdb.Point | kdb.DPoint],
) -> list[kdb.DPoint]:
    return [
        point
        if isinstance(point, kdb.DPoint)
        else (
            kdb.DPoint(point[0], point[1])
            if isinstance(point, tuple)
            else kdb.DPoint(point.x, point.y)
        )
        for point in points
    ]


def spline_points(
    points: np.ndarray | list[tuple[float, float]],
    degree: int = 3,
    npoints: int = 100,
    bc_type: str | None = None,
) -> np.ndarray:
    """Returns smooth points from control points using a B-spline.

    Args:
        points: control points.
        degree: spline degree.
        npoints: number of points to generate.
        bc_type: boundary conditions (e.g., 'clamped', 'natural').

    Example:
        >>> points = [(0, 0), (10, 5), (20, 0)]
        >>> pts = spline_points(points, npoints=50)
    """
    from scipy.interpolate import make_interp_spline

    points = np.asarray(points)
    t = np.linspace(0, 1, len(points))
    k = min(degree, len(points) - 1)
    spl = make_interp_spline(t, points, k=k, bc_type=bc_type)
    t_new = np.linspace(0, 1, npoints)
    return spl(t_new)


def spline_polygon(
    points: np.ndarray | list[tuple[float, float]],
    degree: int = 3,
    npoints: int = 100,
    bc_type: str | None = None,
) -> kdb.DPolygon:
    """Returns a klayout.db.DPolygon from spline-interpolated points.

    Args:
        points: control points.
        degree: spline degree.
        npoints: number of points to generate.
        bc_type: boundary conditions.
    """
    pts = spline_points(points, degree=degree, npoints=npoints, bc_type=bc_type)
    return kdb.DPolygon([kdb.DPoint(p[0], p[1]) for p in pts])


def is_component_spec(obj: Any) -> TypeGuard[ComponentSpec]:
    return isinstance(obj, str | Callable | dict | kf.DKCell)

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.config import ErrorType
from gdsfactory.functions import angles_deg, curvature, snap_angle
from gdsfactory.typings import Coordinate, Coordinates, CrossSectionSpec, Size


def bezier_curve(
    t: npt.NDArray[np.floating[Any]], control_points: Coordinates
) -> npt.NDArray[np.floating[Any]]:
    """Returns bezier coordinates.

    Args:
        t: 1D array of points varying between 0 and 1.
        control_points: for the bezier curve.
    """
    from scipy.special import binom  # type: ignore

    xs = 0.0
    ys = 0.0
    n = len(control_points) - 1
    for k in range(n + 1):
        ank = binom(n, k) * (1 - t) ** (n - k) * t**k
        xs += ank * control_points[k][0]
        ys += ank * control_points[k][1]

    return np.column_stack([xs, ys])


@gf.cell
def bezier(
    control_points: Coordinates = ((0.0, 0.0), (5.0, 0.0), (5.0, 1.8), (10.0, 1.8)),
    npoints: int = 201,
    with_manhattan_facing_angles: bool = True,
    start_angle: int | None = None,
    end_angle: int | None = None,
    cross_section: CrossSectionSpec = "strip",
    bend_radius_error_type: ErrorType | None = None,
    allow_min_radius_violation: bool = False,
) -> Component:
    """Returns Bezier bend.

    Args:
        control_points: list of points.
        npoints: number of points varying between 0 and 1.
        with_manhattan_facing_angles: bool.
        start_angle: optional start angle in deg.
        end_angle: optional end angle in deg.
        cross_section: spec.
        bend_radius_error_type: error type.
        allow_min_radius_violation: bool.
    """
    xs = gf.get_cross_section(cross_section)
    t = np.linspace(0, 1, npoints)
    path_points = bezier_curve(t, control_points)
    path = gf.Path(path_points)

    if with_manhattan_facing_angles:
        path.start_angle = start_angle or snap_angle(path.start_angle)  # type: ignore
        path.end_angle = end_angle or snap_angle(path.end_angle)  # type: ignore

    c = path.extrude(xs)
    curv = curvature(path_points, t)
    length = path.length()
    if max(np.abs(curv)) == 0:
        min_bend_radius = np.inf
    else:
        min_bend_radius = float(gf.snap.snap_to_grid(float(1 / np.max(np.abs(curv)))))

    c.info["length"] = length
    c.info["min_bend_radius"] = min_bend_radius
    c.info["start_angle"] = float(path.start_angle)
    c.info["end_angle"] = float(path.end_angle)
    c.add_route_info(
        cross_section=xs,
        length=c.info["length"],
        n_bend_s=1,
        min_bend_radius=min_bend_radius,
    )

    if not allow_min_radius_violation:
        xs.validate_radius(min_bend_radius, bend_radius_error_type)

    xs.add_bbox(c)
    return c


def find_min_curv_bezier_control_points(
    start_point: Coordinate,
    end_point: Coordinate,
    start_angle: float,
    end_angle: float,
    npoints: int = 201,
    alpha: float = 0.05,
    nb_pts: int = 2,
) -> Coordinates:
    """Returns bezier control points that minimize curvature.

    Args:
        start_point: start point.
        end_point: end point.
        start_angle: start angle in deg.
        end_angle: end angle in deg.
        npoints: number of points varying between 0 and 1.
        alpha: weight for angle mismatch.
        nb_pts: number of control points.
    """
    from scipy.optimize import minimize  # type: ignore

    t = np.linspace(0, 1, npoints)

    def array_1d_to_cpts(a: npt.NDArray[np.float64]) -> list[tuple[float, float]]:
        xs = a[::2]
        ys = a[1::2]
        return list(zip(xs, ys))

    def objective_func(p: npt.NDArray[np.float64]) -> float:
        """Minimize  max curvaturea and negligible start angle and end angle mismatch."""
        ps = array_1d_to_cpts(p)
        control_points = [start_point] + ps + [end_point]
        path_points = bezier_curve(t, control_points)

        max_curv = max(np.abs(curvature(path_points, t)))

        angles = angles_deg(path_points)
        dstart_angle = abs(angles[0] - start_angle)
        dend_angle = abs(angles[-2] - end_angle)
        angle_mismatch = dstart_angle + dend_angle
        return angle_mismatch * alpha + max_curv  # type: ignore

    x0, y0 = start_point[0], start_point[1]
    xn, yn = end_point[0], end_point[1]

    initial_guess: list[float] = []
    for i in range(nb_pts):
        x = (i + 1) * (x0 + xn) / nb_pts
        y = (i + 1) * (y0 + yn) / nb_pts
        initial_guess += [x, y]

    # initial_guess = [(x0 + xn) / 2, y0, (x0 + xn) / 2, yn]
    res = minimize(objective_func, initial_guess, method="Nelder-Mead")  # type: ignore
    p = res.x  # type: ignore
    points = [tuple(start_point)] + array_1d_to_cpts(p) + [tuple(end_point)]  # type: ignore
    return tuple(points)  # type: ignore


@gf.cell
def bend_s(
    size: Size = (11.0, 1.8),
    npoints: int = 99,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
) -> Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction.
        npoints: number of points.
        cross_section: spec.
        allow_min_radius_violation: bool.

    """
    dx, dy = size

    if dy == 0:
        return gf.components.straight(length=dx, cross_section=cross_section)

    return bezier(
        control_points=((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy)),
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )


def get_min_sbend_size(
    size: tuple[float | None, float | None] = (None, 10.0),
    cross_section: CrossSectionSpec = "strip",
    num_points: int = 100,
) -> float:
    """Returns the minimum sbend size to comply with bend radius requirements.

    Args:
        size: in x and y direction. One of them is None, which is the size we need to figure out.
        cross_section: spec.
        num_points: number of points to iterate over between max_size and 0.1 * max_size.
    """
    size_list = list(size)
    cross_section_f = gf.get_cross_section(cross_section)

    if size_list[0] is None:
        ind = 0
        known_s = size_list[1]
    elif size_list[1] is None:
        ind = 1
        known_s = size_list[0]
    else:
        raise ValueError("One of the two elements in size has to be None")

    min_radius = cross_section_f.radius

    if min_radius is None:
        raise ValueError("The min radius for the specified layer is not known!")

    min_size = np.inf

    assert known_s is not None

    # Guess sizes, iterate over them until we cannot achieve the min radius
    # the max size corresponds to an ellipsoid
    max_size = 2.5 * np.sqrt(np.abs(min_radius * known_s))
    sizes: Iterable[float] = np.linspace(max_size, 0.1 * max_size, num_points)  # type: ignore

    assert isinstance(sizes, Iterable)

    for s in sizes:
        sz = size_list
        sz[ind] = s
        dx, dy = size_list
        assert dx is not None and dy is not None
        control_points = ((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy))
        npoints = 201
        t = np.linspace(0, 1, npoints)
        path_points = bezier_curve(t, control_points)
        curv = curvature(path_points, t)
        min_bend_radius = 1 / max(np.abs(curv))
        if min_bend_radius < min_radius:
            min_size = s
            break

    return min_size


if __name__ == "__main__":
    min_size = get_min_sbend_size()
    print(min_size)
    # c = bend_s(size=(10, 0))
    # c = bend_s(bbox_offsets=[0.5], bbox_layers=[(111, 0)], width=2)
    # c = bend_s(size=[10, 2.5])  # 10um bend radius
    # c = bend_s(size=[20, 3], cross_section="rib")  # 10um bend radius
    # c.pprint()
    # c = bend_s_biased()
    # print(c.info["min_bend_radius"])
    # c.show()

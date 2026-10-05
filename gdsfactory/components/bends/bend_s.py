from __future__ import annotations

__all__ = ["bend_s", "bend_s_offset", "bezier"]

import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions.bends.bend_s import bezier_curve
from gdsfactory.config import ErrorType
from gdsfactory.functions import angles_deg, curvature
from gdsfactory.typings import Coordinate, Coordinates, CrossSectionSpec, Size

from .._schematic import sbend_schematic


@gf.cell_with_module_name(schematic_function=sbend_schematic, tags=["bends"])
def bezier(
    control_points: Coordinates = ((0.0, 0.0), (5.0, 0.0), (5.0, 1.8), (10.0, 1.8)),
    npoints: int = 201,
    with_manhattan_facing_angles: bool = True,
    start_angle: int | None = None,
    end_angle: int | None = None,
    cross_section: CrossSectionSpec = "strip",
    bend_radius_error_type: ErrorType | None = None,
    allow_min_radius_violation: bool = False,
    width: float | None = None,
    width_function: gf.typings.WidthFunction | None = None,
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
        width: width to use. Defaults to cross_section.width.
        width_function: optional main-strip width along the extruded path.
    """
    return cf.bezier(
        control_points=control_points,
        npoints=npoints,
        with_manhattan_facing_angles=with_manhattan_facing_angles,
        start_angle=start_angle,
        end_angle=end_angle,
        cross_section=cross_section,
        bend_radius_error_type=bend_radius_error_type,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
        width_function=width_function,
    )


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
    from scipy.optimize import minimize

    t = np.linspace(0, 1, npoints)

    def array_1d_to_cpts(a: npt.NDArray[np.float64]) -> list[tuple[float, float]]:
        xs = a[::2]
        ys = a[1::2]
        return list(zip(xs, ys, strict=False))

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
        return float(angle_mismatch * alpha + max_curv)

    x0, y0 = start_point[0], start_point[1]
    xn, yn = end_point[0], end_point[1]

    initial_guess: list[float] = []
    for i in range(nb_pts):
        x = (i + 1) * (x0 + xn) / nb_pts
        y = (i + 1) * (y0 + yn) / nb_pts
        initial_guess += [x, y]

    # initial_guess = [(x0 + xn) / 2, y0, (x0 + xn) / 2, yn]
    res = minimize(objective_func, initial_guess, method="Nelder-Mead")
    p = res.x
    points = [start_point] + array_1d_to_cpts(p) + [end_point]
    return tuple(points)


@gf.cell_with_module_name(schematic_function=sbend_schematic, tags=["bends"])
def bend_s(
    size: Size = (11.0, 1.8),
    npoints: int = 99,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    width: float | None = None,
) -> Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction.
        npoints: number of points.
        cross_section: spec.
        allow_min_radius_violation: bool.
        width: width to use. Defaults to cross_section.width.

    """
    return cf.bend_s(
        size=size,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
    )


@gf.cell_with_module_name(schematic_function=sbend_schematic, tags=["bends"])
def bend_s_offset(
    offset: float = 40.0,
    radius: float | None = 10.0,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
    with_euler: bool | None = None,
    p: float = 1,
    with_arc_floorplan: bool = False,
    npoints: int | None = None,
    angular_step: float | None = None,
) -> gf.Component:
    """Return S bend made of two bends with a straight section.

    Args:
        offset: in um.
        radius: in um. if None, uses cross_section_radius.
        cross_section: spec.
        width: width to use. Defaults to cross_section.width.
        with_euler: deprecated, use p=0 for circular arc instead.
        p: 1 means standard Euler bend. 0 means circular arc.
        with_arc_floorplan: if True the size of the bend will be adjusted to match an arc bend with the specified radius. If False: `radius` is the minimum radius of curvature.
        npoints: number of points.
        angular_step: If provided, determines the angular step (in degrees) between points. Mutually exclusive with npoints.
    """
    return cf.bend_s_offset(
        offset=offset,
        radius=radius,
        cross_section=cross_section,
        width=width,
        with_euler=with_euler,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        angular_step=angular_step,
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
    max_size = float(np.sqrt(np.abs(min_radius * known_s)) * 2.5)
    sizes = np.linspace(max_size, 0.1 * max_size, num_points)

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

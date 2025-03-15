from __future__ import annotations

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
    from scipy.special import binom

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
    width: float | None = None,
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
    """
    if width:
        xs = gf.get_cross_section(cross_section, width=width)
    else:
        xs = gf.get_cross_section(cross_section)

    t = np.linspace(0, 1, npoints)
    path_points = bezier_curve(t, control_points)
    path = gf.Path(path_points)

    if with_manhattan_facing_angles:
        path.start_angle = start_angle or snap_angle(path.start_angle)
        path.end_angle = end_angle or snap_angle(path.end_angle)

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
    from scipy.optimize import minimize

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


@gf.cell
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
    dx, dy = size

    if dy == 0:
        return gf.components.straight(
            length=dx, cross_section=cross_section, width=width
        )

    return bezier(
        control_points=((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy)),
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
    )


def _get_arc_sbend_angle_middle_length_from_jog(
    jog: float, radius: float
) -> tuple[float, float]:
    """Compute the Euler bend angle and middle straight length for an S-bend."""
    if jog < 2 * radius:
        angle = np.rad2deg(2 * np.arcsin(np.sqrt(jog / (4 * radius))))
        middle_length = 0.0
    else:
        angle = 90.0
        middle_length = jog - 2 * radius
    return (angle, middle_length)


def _get_euler_sbend_angle_middle_length_from_jog(
    jog: float, radius: float
) -> tuple[float, float]:
    """Compute the Euler bend angle (in degrees) and middle straight length for an S-bend.

    using SciPy to numerically solve for the bend angle required to achieve half the jog.

    The vertical displacement for an Euler bend of angle θ (in radians) is given by:
      displacement(θ) = radius * sqrt(pi * θ) * S( sqrt(2θ/pi) )
    where S() is the Fresnel sine integral.

    The S-bend consists of two symmetric Euler bends. If the jog is less than twice the displacement
    of a full 90° Euler bend, the angle is computed such that one Euler bend gives a displacement of jog/2.
    Otherwise, a full 90° Euler bend is used and the extra required offset is added as a straight section.

    Args:
        jog: The vertical displacement of the S-bend.
        radius: The radius of the Euler bend.

    Returns:
      tuple: (angle_deg, middle_length) where:
          - angle_deg is the Euler bend angle in degrees.
          - middle_length is the length of the straight segment between the Euler bends.
    """
    from scipy import optimize

    def euler_displacement(theta: float) -> float:
        curve = gf.path.euler(radius=radius, angle=theta, use_eff=False, p=1)
        return curve.ysize

    dy_full = euler_displacement(theta=90)

    if jog < 2 * dy_full:
        # Define the objective function: squared error between computed displacement and jog
        def objective(theta: float) -> float:
            return (euler_displacement(theta) - jog) ** 2

        result = optimize.minimize_scalar(objective, bounds=(1, 90), method="bounded")
        angle_deg = result.x
        middle_length = 0.0
    else:
        angle_deg = 90.0
        middle_length = jog - 2 * dy_full

    return angle_deg, middle_length


@gf.cell
def bend_s_offset(
    offset: float = 40.0,
    radius: float = 10.0,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
    with_euler: bool = True,
) -> gf.Component:
    """Return S bend made of two euler bends with a straight section.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        offset: in um.
        radius: in um.
        cross_section: spec.
        width: width to use. Defaults to cross_section.width.
        with_euler: use euler bend instead of arc bend.
    """
    if width:
        xs = gf.get_cross_section(cross_section, width=width)
    else:
        xs = gf.get_cross_section(cross_section)

    xs.validate_radius(radius)
    if with_euler:
        angle, middle_length = _get_euler_sbend_angle_middle_length_from_jog(
            jog=offset / 2, radius=radius
        )
        path = gf.path.euler(radius=radius, angle=+angle, p=1, use_eff=False)
        path += gf.path.straight(length=middle_length)
        path += gf.path.euler(radius=radius, angle=-angle, p=1, use_eff=False)
    else:
        angle, middle_length = _get_arc_sbend_angle_middle_length_from_jog(
            jog=offset,
            radius=radius,
        )
        path = gf.path.arc(radius=radius, angle=+angle)
        path += gf.path.straight(length=middle_length)
        path += gf.path.arc(radius=radius, angle=-angle)

    return gf.path.extrude(path, cross_section=xs)


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


if __name__ == "__main__":
    # xs = gf.cross_section.strip(width=2)
    # c = bend_s_offset(offset=40, with_arc_floorplan=False, cross_section=xs, width=1)
    # print(c.info["min_bend_radius"])
    c = bend_s_offset(offset=1, radius=10, with_euler=True)
    c.show()

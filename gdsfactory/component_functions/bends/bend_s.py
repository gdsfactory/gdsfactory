from __future__ import annotations

from gdsfactory.cross_section.utils import validate_radius

__all__ = ["bend_s", "bend_s_offset", "bezier"]

import math
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.component_functions._get_component import get_component
from gdsfactory.config import ErrorType
from gdsfactory.functions import curvature, snap_angle
from gdsfactory.typings import Coordinates, CrossSectionSpec, Size


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

    c = path.extrude(xs, width_function=width_function, add_bbox=True)
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
        validate_radius(xs, min_bend_radius, bend_radius_error_type)

    return c


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
        return get_component(
            "straight", length=dx, cross_section=cross_section, width=width
        )

    return get_component(
        "bezier",
        control_points=((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy)),
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
    )


def _get_euler_sbend_angle_middle_length_from_jog(
    jog: float, radius: float, p: float = 1, use_eff: bool = False
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
        p: proportion of the curve that is an Euler curve.
        use_eff: if True, use effective radius.

    Returns:
      tuple: (angle_deg, middle_length) where:
          - angle_deg is the Euler bend angle in degrees.
          - middle_length is the length of the straight segment between the Euler bends.
    """
    from scipy import optimize

    def euler_displacement(theta: float) -> float:
        curve = gf.path.euler(radius=radius, angle=theta, use_eff=use_eff, p=p)
        return curve.ysize

    dy_full = euler_displacement(theta=90)

    if jog <= dy_full:
        # Define the objective function: squared error between computed displacement and jog
        def objective(theta: float) -> float:
            return (euler_displacement(theta) - jog) ** 2

        result = optimize.minimize_scalar(objective, bounds=(1, 90), method="bounded")
        angle_deg = result.x
        middle_length = 0.0
    else:
        angle_deg = 90.0
        middle_length = 2 * jog - 2 * dy_full

    return angle_deg, middle_length


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
    if with_euler is not None:
        warnings.warn(
            "with_euler is deprecated. Use p=0 for circular arc instead. And p=1 for euler bend.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not with_euler:
            p = 0

    if width:
        xs = gf.get_cross_section(cross_section, width=width)
    else:
        xs = gf.get_cross_section(cross_section)

    radius = radius or xs.radius
    assert radius is not None, "radius cannot be None"

    validate_radius(xs, radius)
    angle, middle_length = _get_euler_sbend_angle_middle_length_from_jog(
        jog=abs(offset) / 2, radius=radius, p=p, use_eff=with_arc_floorplan
    )
    angle = math.copysign(angle, offset)
    path = gf.path.euler(
        radius=radius,
        angle=+angle,
        p=p,
        use_eff=with_arc_floorplan,
        npoints=npoints,
        angular_step=angular_step,
    )
    if middle_length > 1e-6:
        path += gf.path.straight(length=middle_length)
    path += gf.path.euler(
        radius=radius,
        angle=-angle,
        p=p,
        use_eff=with_arc_floorplan,
        npoints=npoints,
        angular_step=angular_step,
    )

    return gf.path.extrude(path, cross_section=xs)

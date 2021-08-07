from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
from scipy.special import binom

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.config import TECH
from gdsfactory.geo_utils import (
    angles_deg,
    curvature,
    extrude_path,
    path_length,
    snap_angle,
)
from gdsfactory.hash_points import hash_points
from gdsfactory.types import Coordinate, Coordinates, Number


def bezier_curve(t: ndarray, control_points: Coordinates) -> ndarray:
    """Returns bezier coordinates

    Args:
        t: 1D array of points varying between 0 and 1.
        control_points:
    """
    xs = 0.0
    ys = 0.0
    n = len(control_points) - 1
    for k in range(n + 1):
        ank = binom(n, k) * (1 - t) ** (n - k) * t ** k
        xs += ank * control_points[k][0]
        ys += ank * control_points[k][1]

    return np.column_stack([xs, ys])


def bezier_points(control_points: Coordinates, width: Number, npoints: int = 101):
    t = np.linspace(0, 1, npoints)
    points = bezier_curve(t, control_points)
    return extrude_path(points, width)


def bezier_biased(width=0.5, **kwargs):
    width = gf.bias.width(width)
    return bezier(width=width, **kwargs)


@gf.cell
def bezier(
    name: Optional[str] = None,
    width: float = TECH.waveguide.strip.width,
    control_points: Coordinates = ((0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (10.0, 2.0)),
    npoints: int = 201,
    layer: Tuple[int, int] = TECH.waveguide.strip.layer,
    with_manhattan_facing_angles: bool = True,
    spike_length: float = 0.0,
    start_angle: Optional[int] = None,
    end_angle: Optional[int] = None,
    grid: float = 0.001,
) -> Component:
    """Bezier bend
    We avoid autoname for control_points

    Args:
        name: for the cell
        width: straight width
        control_points: list of points
        npoints: number of points varying between 0 and 1
        layer: layer
        with_manhattan_facing_angles:
        spike_length
        start_angle
        end_angle
        grid
    """

    if name is None:
        points_hash = hash_points(control_points)
        name = f"bezier_w{int(width*1e3)}_{points_hash}_{layer[0]}_{layer[1]}"

    c = gf.Component(name=name)
    c.ignore.add("control_points")
    t = np.linspace(0, 1, npoints)
    path_points = bezier_curve(t, control_points)
    polygon_points = extrude_path(
        path_points,
        width=width,
        with_manhattan_facing_angles=with_manhattan_facing_angles,
        spike_length=spike_length,
        start_angle=start_angle,
        end_angle=end_angle,
        grid=grid,
    )
    angles = angles_deg(path_points)

    c.info["start_angle"] = gf.snap.snap_to_grid(angles[0])
    c.info["end_angle"] = gf.snap.snap_to_grid(angles[-2])

    a0 = angles[0] + 180
    a1 = angles[-2]

    a0 = snap_angle(a0)
    a1 = snap_angle(a1)

    p0 = path_points[0]
    p1 = path_points[-1]
    c.add_polygon(polygon_points, layer=layer)
    c.add_port(name="0", midpoint=p0, width=width, orientation=a0, layer=layer)
    c.add_port(name="1", midpoint=p1, width=width, orientation=a1, layer=layer)

    curv = curvature(path_points, t)
    length = gf.snap.snap_to_grid(path_length(path_points))
    min_bend_radius = gf.snap.snap_to_grid(1 / max(np.abs(curv)))
    c.info["length"] = length
    c.info["min_bend_radius"] = min_bend_radius
    # c.info["curvature"] = curv
    # c.info["t"] = t
    c.length = length
    c.min_bend_radius = min_bend_radius
    return c


def find_min_curv_bezier_control_points(
    start_point: ndarray,
    end_point: Coordinate,
    start_angle: int,
    end_angle: int,
    npoints: int = 201,
    alpha: float = 0.05,
    nb_pts: int = 2,
) -> Coordinates:
    t = np.linspace(0, 1, npoints)

    def array_1d_to_cpts(a):
        xs = a[::2]
        ys = a[1::2]
        return [(x, y) for x, y in zip(xs, ys)]

    def objective_func(p):
        """
        We want to minimize a combination of:
            - max curvature
            - negligible mismatch with start angle and end angle
        """

        ps = array_1d_to_cpts(p)
        control_points = [start_point] + ps + [end_point]
        path_points = bezier_curve(t, control_points)

        max_curv = max(np.abs(curvature(path_points, t)))

        angles = angles_deg(path_points)
        dstart_angle = abs(angles[0] - start_angle)
        dend_angle = abs(angles[-2] - end_angle)
        angle_mismatch = dstart_angle + dend_angle
        return angle_mismatch * alpha + max_curv

    x0, y0 = start_point[0], start_point[1]
    xn, yn = end_point[0], end_point[1]

    initial_guess = []
    for i in range(nb_pts):
        x = (i + 1) * (x0 + xn) / (nb_pts)
        y = (i + 1) * (y0 + yn) / (nb_pts)
        initial_guess += [x, y]

    # initial_guess = [(x0 + xn) / 2, y0, (x0 + xn) / 2, yn]

    res = minimize(objective_func, initial_guess, method="Nelder-Mead")

    p = res.x
    return [tuple(start_point)] + array_1d_to_cpts(p) + [tuple(end_point)]


if __name__ == "__main__":
    control_points = ((0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (10.0, 5.0))
    c = bezier(control_points=control_points)
    c.pprint()
    # print(c.ports)
    # print(c.ports["0"].y - c.ports["1"].y)
    # print(c.ignore)
    # c.write_gds()
    c.show()

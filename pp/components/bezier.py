from typing import List, Tuple
import hashlib
import numpy as np
from numpy import ndarray
from scipy.special import binom
from scipy.optimize import minimize

import pp
from pp.layers import LAYER
from pp.geo_utils import extrude_path
from pp.geo_utils import angles_deg
from pp.geo_utils import snap_angle
from pp.geo_utils import path_length
from pp.geo_utils import curvature
from pp.component import Component


def bezier_curve(t: ndarray, control_points: List[Tuple[float, int]]) -> ndarray:
    xs = 0.0
    ys = 0.0
    n = len(control_points) - 1
    for k in range(n + 1):
        ank = binom(n, k) * (1 - t) ** (n - k) * t ** k
        xs += ank * control_points[k][0]
        ys += ank * control_points[k][1]

    return np.column_stack([xs, ys])


def bezier_points(control_points, width, t=np.linspace(0, 1, 101)):
    """t: 1D array of points varying between 0 and 1"""
    points = bezier_curve(t, control_points)
    return extrude_path(points, width)


def bezier_biased(width=0.5, **kwargs):
    width = pp.bias.width(width)
    return bezier(width=width, **kwargs)


# Not using autoname on bezier due to control_points and t spacing
def bezier(
    name: None = None,
    width: float = 0.5,
    control_points: List[Tuple[float, float]] = [
        (0.0, 0.0),
        (5.0, 0.0),
        (5.0, 2.0),
        (10.0, 2.0),
    ],
    t: ndarray = np.linspace(0, 1, 201),
    layer: Tuple[int, int] = LAYER.WG,
    **extrude_path_params,
) -> Component:
    """ bezier bend """

    def format_float(x):
        return "{:.3f}".format(x).rstrip("0").rstrip(".")

    def _fmt_cp(cps):
        return "_".join([f"({format_float(p[0])},{format_float(p[1])})" for p in cps])

    if name is None:
        points_hash = hashlib.md5(_fmt_cp(control_points).encode()).hexdigest()
        name = f"bezier_w{int(width*1e3)}_{points_hash}_{layer[0]}_{layer[1]}"

    c = pp.Component(name=name)
    path_points = bezier_curve(t, control_points)
    polygon_points = extrude_path(path_points, width, **extrude_path_params)
    angles = angles_deg(path_points)

    c.info["start_angle"] = angles[0]
    c.info["end_angle"] = angles[-2]

    a0 = angles[0] + 180
    a1 = angles[-2]

    a0 = snap_angle(a0)
    a1 = snap_angle(a1)

    p0 = path_points[0]
    p1 = path_points[-1]
    c.add_polygon(polygon_points, layer=layer)
    c.add_port(name="0", midpoint=p0, width=width, orientation=a0, layer=layer)
    c.add_port(name="1", midpoint=p1, width=width, orientation=a1, layer=layer)

    c.info["length"] = path_length(path_points)
    curv = curvature(path_points, t)
    c.info["min_bend_radius"] = 1 / max(np.abs(curv))
    c.info["curvature"] = curv
    c.info["t"] = t

    return c


def find_min_curv_bezier_control_points(
    start_point,
    end_point,
    start_angle,
    end_angle,
    t=np.linspace(0, 1, 201),
    alpha=0.05,
    nb_pts=2,
):
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
    return [start_point] + array_1d_to_cpts(p) + [end_point]


if __name__ == "__main__":
    c = bezier()
    print(c.ports)
    print(c.ports["0"].y - c.ports["1"].y)
    pp.write_gds(c)
    pp.show(c)

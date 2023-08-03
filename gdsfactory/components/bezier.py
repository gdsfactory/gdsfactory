from __future__ import annotations

import numpy as np
from numpy import ndarray

import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.component import Component
from gdsfactory.geometry.functions import angles_deg, curvature, path_length, snap_angle
from gdsfactory.typings import Coordinate, Coordinates, CrossSectionSpec


def bezier_curve(t: ndarray, control_points: Coordinates) -> ndarray:
    """Returns bezier coordinates.

    Args:
        t: 1D array of points varying between 0 and 1.
        control_points:
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
    control_points: Coordinates = ((0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (10.0, 2.0)),
    npoints: int = 201,
    with_manhattan_facing_angles: bool = True,
    start_angle: int | None = None,
    end_angle: int | None = None,
    cross_section: CrossSectionSpec = "strip",
    with_bbox: bool = True,
    **kwargs,
) -> Component:
    """Returns Bezier bend.

    Args:
        control_points: list of points.
        npoints: number of points varying between 0 and 1.
        with_manhattan_facing_angles: bool.
        start_angle: optional start angle in deg.
        end_angle: optional end angle in deg.
        cross_section: spec.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
    """
    xs = gf.get_cross_section(cross_section, **kwargs)
    t = np.linspace(0, 1, npoints)
    path_points = bezier_curve(t, control_points)
    path = gf.Path(path_points)

    if with_manhattan_facing_angles:
        path.start_angle = start_angle or snap_angle(path.start_angle)
        path.end_angle = end_angle or snap_angle(path.end_angle)

    c = Component()
    bend = path.extrude(xs)
    bend_ref = c << bend
    c.add_ports(bend_ref.ports)
    c.absorb(bend_ref)
    curv = curvature(path_points, t)
    length = gf.snap.snap_to_grid(path_length(path_points))
    if max(np.abs(curv)) == 0:
        min_bend_radius = np.inf
    else:
        min_bend_radius = gf.snap.snap_to_grid(1 / max(np.abs(curv)))

    c.info["length"] = length
    c.info["min_bend_radius"] = min_bend_radius
    c.info["start_angle"] = path.start_angle
    c.info["end_angle"] = path.end_angle

    if with_bbox and xs.bbox_layers:
        padding = []
        for offset in xs.bbox_offsets:
            points = get_padding_points(
                component=c,
                default=0,
                bottom=offset,
                top=offset,
            )
            padding.append(points)

        for layer, points in zip(xs.bbox_layers, padding):
            c.add_polygon(points, layer=layer)
    return c


def find_min_curv_bezier_control_points(
    start_point: ndarray,
    end_point: Coordinate,
    start_angle: float,
    end_angle: float,
    npoints: int = 201,
    alpha: float = 0.05,
    nb_pts: int = 2,
) -> Coordinates:
    from scipy.optimize import minimize

    t = np.linspace(0, 1, npoints)

    def array_1d_to_cpts(a):
        xs = a[::2]
        ys = a[1::2]
        return list(zip(xs, ys))

    def objective_func(p):
        """We want to minimize a combination of the following.

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
        x = (i + 1) * (x0 + xn) / nb_pts
        y = (i + 1) * (y0 + yn) / nb_pts
        initial_guess += [x, y]

    # initial_guess = [(x0 + xn) / 2, y0, (x0 + xn) / 2, yn]

    res = minimize(objective_func, initial_guess, method="Nelder-Mead")

    p = res.x
    return [tuple(start_point)] + array_1d_to_cpts(p) + [tuple(end_point)]


if __name__ == "__main__":
    # control_points = ((0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (10.0, 5.0))
    # cross_section = gf.get_cross_section(
    #     "strip", bbox_offsets=[0.5], bbox_layers=[(111, 0)]
    # )
    # npoints = 201
    # with_manhattan_facing_angles = True
    # xs = gf.get_cross_section(
    #     cross_section,
    # )
    # t = np.linspace(0, 1, npoints)
    # path_points = bezier_curve(t, control_points)
    # path = gf.Path(path_points)

    # if with_manhattan_facing_angles:
    #     angles = angles_deg(path_points)
    #     path.start_angle = snap_angle(angles[0])
    #     path.end_angle = snap_angle(angles[-2])
    # c = path.extrude(xs)

    # points = ((0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (10.0, 5.0))
    # c = bezier(control_points=points)
    # c.pprint()
    # print(c.ports)
    # print(c.ports["0"].y - c.ports["1"].y)
    # c.write_gds()

    c = bezier(bbox_offsets=[0.5], bbox_layers=[(111, 0)])
    c.show(show_ports=True)

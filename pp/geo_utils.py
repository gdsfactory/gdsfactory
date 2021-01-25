from typing import List, Optional, Union

import numpy as np
from numpy import cos, float64, ndarray, sin

from pp.coord2 import Coord2

RAD2DEG = 180.0 / np.pi
DEG2RAD = 1 / RAD2DEG


def sign_shape(pts: ndarray) -> float64:
    pts2 = np.roll(pts, 1, axis=0)
    dx = pts2[:, 0] - pts[:, 0]
    y = pts2[:, 1] + pts[:, 1]
    return np.sign((dx * y).sum())


def area(pts: ndarray) -> float64:
    pts2 = np.roll(pts, 1, axis=0)
    dx = pts2[:, 0] - pts[:, 0]
    y = pts2[:, 1] + pts[:, 1]
    return (dx * y).sum() / 2


def manhattan_direction(p0, p1, tol=1e-5):
    """
    """
    dp = p1 - p0
    dx, dy = dp[0], dp[1]
    if abs(dx) < tol:
        sx = 0
    elif dx > 0:
        sx = 1
    else:
        sx = -1

    if abs(dy) < tol:
        sy = 0
    elif dy > 0:
        sy = 1
    else:
        sy = -1
    return np.array((sx, sy))


def remove_flat_angles(points: ndarray) -> ndarray:
    a = angles_deg(np.vstack(points))
    da = a - np.roll(a, 1)
    da = np.mod(np.round(da, 3), 180)

    # To make sure we do not remove points at the edges
    da[0] = 1
    da[-1] = 1

    to_rm = list(np.where(np.abs(da[:-1]) < 1e-9)[0])
    if isinstance(points, list):
        while to_rm:
            i = to_rm.pop()
            points.pop(i)

    else:
        points = points[da != 0]

    return points


def remove_identicals(
    pts: ndarray, grids_per_unit: int = 1000, closed: bool = True
) -> ndarray:
    if len(pts) > 1:
        identicals = np.prod(abs(pts - np.roll(pts, -1, 0)) < 0.5 / grids_per_unit, 1)
        if not closed:
            identicals[-1] = False
        pts = np.delete(pts, identicals.nonzero()[0], 0)
    return pts


def centered_diff(a: ndarray) -> ndarray:
    d = (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / 2
    return d[1:-1]


def centered_diff2(a: ndarray) -> ndarray:
    d = (np.roll(a, -1, axis=0) - a) - (a - np.roll(a, 1, axis=0))
    return d[1:-1]


def curvature(points: ndarray, t: ndarray) -> ndarray:
    """

    Args:
        points : numpy.array shape (n, 2)
        t: numpy.array of size n

    Return:
        The curvature at each point

    Computes the curvature at every point excluding the first and last point

    For a planar curve parametrized as P(t) = (x(t), y(t)), the curvature is given
    by (x' y'' - x'' y' ) / (x' **2 + y' **2)**(3/2)

    """

    # Use centered difference for derivative
    dt = centered_diff(t)
    dp = centered_diff(points)
    dp2 = centered_diff2(points)

    dx = dp[:, 0] / dt
    dy = dp[:, 1] / dt

    dx2 = dp2[:, 0] / dt ** 2
    dy2 = dp2[:, 1] / dt ** 2

    curv = (dx * dy2 - dx2 * dy) / (dx ** 2 + dy ** 2) ** (3 / 2)
    return curv


def radius_of_curvature(points, t):
    return 1 / curvature(points, t)


def path_length(points: ndarray) -> float64:
    """
    Args:
        points: <np.array>
            With shape (N, 2) representing N points with coordinates x, y

    Returns:
        <float> The path length
    """

    dpts = points[1:, :] - points[:-1, :]
    _d = dpts ** 2
    return np.sum(np.sqrt(_d[:, 0] + _d[:, 1]))


def snap_angle(a: float64) -> int:
    """
    a: angle in deg
    Return angle snapped along manhattan angle
    """
    a = a % 360
    if -45 < a < 45:
        _a = 0
    elif 45 < a < 135:
        _a = 90
    elif 135 < a < 225:
        _a = 180
    elif 225 < a < 315:
        _a = 270
    else:
        _a = 0
    return _a


def angles_rad(pts: ndarray) -> ndarray:
    """ returns the angles (radians) of the connection between each point and the next """
    _pts = np.roll(pts, -1, 0)
    radians = np.arctan2(_pts[:, 1] - pts[:, 1], _pts[:, 0] - pts[:, 0])
    return radians


def angles_deg(pts: ndarray) -> ndarray:
    """ returns the angles (degrees) of the connection between each point and the next """
    return angles_rad(pts) * RAD2DEG


def extrude_path(
    points: Union[List[Coord2], ndarray],
    width: float,
    with_manhattan_facing_angles: bool = True,
    spike_length: Union[float64, int, float] = 0,
    start_angle: Optional[int] = None,
    end_angle: Optional[int] = None,
    grid: float = 0.001,
) -> ndarray:
    """
    Extrude a path of width `width` along a curve defined by `points`

    Args:
        points: numpy 2D array of shape (N, 2)
        width: float
        with_manhattan_facing_angles: bool
        spike_length:
        start_angle:
        end_angle:
        grid:

    Returns:
        numpy 2D array of shape (2*N, 2)
    """

    if isinstance(points, list):
        points = np.stack([(p[0], p[1]) for p in points], axis=0)

    a = angles_deg(points)
    if with_manhattan_facing_angles:
        _start_angle = snap_angle(a[0] + 180)
        _end_angle = snap_angle(a[-2])
    else:
        _start_angle = a[0] + 180
        _end_angle = a[-2]

    start_angle = start_angle if start_angle is not None else _start_angle
    end_angle = end_angle if end_angle is not None else _end_angle

    a2 = angles_rad(points) * 0.5
    a1 = np.roll(a2, 1)

    a2[-1] = end_angle * DEG2RAD - a2[-2]
    a1[0] = start_angle * DEG2RAD - a1[1]

    a_plus = a2 + a1
    cos_a_min = np.cos(a2 - a1)
    offsets = np.column_stack((-sin(a_plus) / cos_a_min, cos(a_plus) / cos_a_min)) * (
        0.5 * width
    )

    points_back = np.flipud(points - offsets)
    if spike_length != 0:
        d = spike_length
        a_start = start_angle * DEG2RAD
        a_end = end_angle * DEG2RAD
        p_start_spike = points[0] + d * np.array([[cos(a_start), sin(a_start)]])
        p_end_spike = points[-1] + d * np.array([[cos(a_end), sin(a_end)]])

        pts = np.vstack((p_start_spike, points + offsets, p_end_spike, points_back))
    else:
        pts = np.vstack((points + offsets, points_back))

    pts = np.round(pts / grid) * grid

    return pts


def polygon_grow(polygon: ndarray, offset: float) -> ndarray:
    """
    polygon has to be a closed shape
    """
    s = remove_identicals(polygon)
    s = remove_flat_angles(s)
    s = np.vstack([s, s[0]])
    if len(s) <= 1:
        return s

    # Make sure the shape is oriented in the correct direction for scaling
    ss = sign_shape(s)
    offset = -ss * offset

    a2 = angles_rad(s) * 0.5
    a1 = np.roll(a2, 1)

    a2[-1] = a2[0]
    a1[0] = a1[-1]

    a = a2 + a1
    c_minus = cos(a2 - a1)
    offsets = np.column_stack((-sin(a) / c_minus, cos(a) / c_minus)) * offset
    # compute offsets from each point

    pts = s + offsets
    return pts

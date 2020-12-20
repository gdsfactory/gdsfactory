from typing import List, Union

import numpy as np
from numpy import pi, sqrt
from scipy.special import fresnel

from pp.coord2 import Coord2

DEG2RAD = np.pi / 180

# Dict for caching Euler bend point lists
__euler_bend_cache__ = dict()


def euler_bend_points(
    angle_amount: float = 90.0,
    radius: float = 10.0,
    resolution: float = 150.0,
    use_cache: bool = True,
) -> List[Coord2]:
    """ Base euler bend, no transformation, emerging from the origin."""
    # Check if we've calculated this already
    key = (angle_amount, radius, resolution)
    if key in __euler_bend_cache__ and use_cache:
        return __euler_bend_cache__[key]

    if angle_amount < 0:
        raise ValueError("angle_amount should be positive. Got {}".format(angle_amount))
    # End angle
    eth = angle_amount * DEG2RAD

    # If bend is trivial, return a trivial shape
    if eth == 0.0:
        return [Coord2(0, 0)]

    # Curve min radius
    R = radius

    # Total displaced angle
    th = eth / 2.0

    # Total length of curve
    Ltot = 4 * R * th

    # Compute curve ##
    a = sqrt(R ** 2.0 * np.abs(th))
    sq2pi = sqrt(2.0 * pi)

    # Function for computing curve coords
    (fasin, facos) = fresnel(sqrt(2.0 / pi) * R * th / a)

    def _xy(s):
        if th == 0:
            return Coord2(0.0, 0.0)
        elif s <= Ltot / 2:
            (fsin, fcos) = fresnel(s / (sq2pi * a))
            X = sq2pi * a * fcos
            Y = sq2pi * a * fsin
        else:
            (fsin, fcos) = fresnel((Ltot - s) / (sq2pi * a))
            X = (
                sq2pi
                * a
                * (
                    facos
                    + np.cos(2 * th) * (facos - fcos)
                    + np.sin(2 * th) * (fasin - fsin)
                )
            )
            Y = (
                sq2pi
                * a
                * (
                    fasin
                    - np.cos(2 * th) * (fasin - fsin)
                    + np.sin(2 * th) * (facos - fcos)
                )
            )
        return Coord2(X, Y)

    # Parametric step size
    step = Ltot / int(th * resolution)

    # Generate points
    points = []
    for i in range(0, int(round(Ltot / step)) + 1):
        points += [_xy(i * step)]

    # Cache calculated points
    if use_cache:
        __euler_bend_cache__[(angle_amount, radius, resolution)] = points

    return points


def euler_end_pt(
    start_point=(0.0, 0.0), radius=10.0, input_angle=0.0, angle_amount=90.0
):
    """Gives the end point of a simple Euler bend as a Coord2"""

    th = abs(angle_amount) * DEG2RAD / 2.0
    R = radius
    clockwise = bool(angle_amount < 0)

    (fsin, fcos) = fresnel(sqrt(2 * th / pi))

    a = 2 * sqrt(2 * pi * th) * (np.cos(th) * fcos + np.sin(th) * fsin)
    r = a * R
    X = r * np.cos(th)
    Y = r * np.sin(th)

    if clockwise:
        Y *= -1

    pt = Coord2(X, Y) + start_point
    pt.rotate(rotation_center=start_point, rotation=input_angle)
    return pt


def euler_length(radius: Union[int, float] = 10.0, angle_amount: int = 90.0) -> float:
    th = abs(angle_amount) * DEG2RAD / 2
    return 4 * radius * th

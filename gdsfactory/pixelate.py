from __future__ import annotations

import itertools as it
from typing import Optional

import numpy as np
from shapely import geometry
from shapely.geometry.polygon import Polygon

from gdsfactory.geometry.functions import polygon_grow
from gdsfactory.typings import Coordinates

DEG2RAD = np.pi / 180
RAD2DEG = 1.0 / DEG2RAD
# from matplotlib import pyplot as plt


def pixelate_path(
    pts: Coordinates,
    pixel_size: float = 0.55,
    snap_res: float = 0.05,
    middle_offset: float = 0.5,
    theta_start: float = 0,
    theta_end: float = 90,
) -> Coordinates:
    """From a path add one pixel per point on the path.

    Args:
        pts: points.
        pixel_size: in um.
        snap_res: snap resolution.
        middle_offset: in um.
        theta_start: in degrees.
        theta_end: in degrees.

    """
    thetas0 = [
        np.arctan2(y1 - y0, x1 - x0) for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:])
    ]
    thetas0 += [theta_end * DEG2RAD]
    thetas0[0] = theta_start * DEG2RAD

    thetas1 = [
        np.arctan2(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:])
    ]
    thetas1[0] = (theta_start - 180) * DEG2RAD
    thetas1 += [(theta_end - 180) * DEG2RAD]

    thetas_deg0 = np.abs(np.array(thetas0) * RAD2DEG) % 90
    thetas_deg1 = np.abs(np.array(thetas1) * RAD2DEG) % 90

    thetas_deg = thetas_deg1

    slice = np.where(thetas_deg0 < 45)
    thetas_deg[slice] = thetas_deg0[slice]

    scalings = np.cos(
        abs(thetas_deg) * DEG2RAD
    )  # + middle_offset * (1 - np.cos(abs(thetas_deg) * DEG2RAD) )

    def _snap(x: float) -> float:
        return round(int(x / snap_res), 0) * snap_res

    def _gen_pixel(p, a):
        x0, y0 = p
        pix = [(x0 + a, y0 - a), (x0 + a, y0 + a), (x0 - a, y0 + a), (x0 - a, y0 - a)]
        pix = [(_snap(_x), _snap(_y)) for _x, _y in pix]
        return pix

    a = pixel_size / 2
    return [_gen_pixel(p, a * s) for p, s in zip(pts, scalings)]


def points_to_shapely(pts: Coordinates) -> Polygon:
    return Polygon(pts)


def _snap_to_resolution(a, snap_res):
    return np.round(a / snap_res, 0) * snap_res


def _pixelate(
    pts: Coordinates,
    N: int = 100,
    margin: float = 0.4,
    margin_x: Optional[float] = None,
    margin_y: Optional[float] = None,
    nb_pixels_x: Optional[int] = None,
    nb_pixels_y: Optional[int] = None,
    min_pixel_size: float = 0.4,
    snap_res: float = 0.05,
) -> Coordinates:
    """Pixelates a shape (as 2d array) onto an NxN grid.

    Arguments:
        pts: The 2D array to be pixelated.
        N: The number of pixels on an edge of the grid.

    Returns:
        A list of pixel bounding boxes

    """
    shape = points_to_shapely(pts)  # convert to shapely
    if not shape:
        return []

    if margin_x is None:
        margin_x = margin

    if margin_y is None:
        margin_y = margin

    if nb_pixels_x is None:
        nb_pixels_x = N

    if nb_pixels_y is None:
        nb_pixels_y = N

    west, south, east, north = shape.bounds

    width = east - west + 2 * margin_x
    height = north - south + 2 * margin_y

    if min_pixel_size is not None:
        max_nb_pixels_x = int(np.ceil(width / min_pixel_size))
        nb_pixels_x = min(nb_pixels_x, max_nb_pixels_x)

    if min_pixel_size is not None:
        max_nb_pixels_y = int(np.ceil(height / min_pixel_size))
        nb_pixels_y = min(nb_pixels_y, max_nb_pixels_y)

    w = width / nb_pixels_x + snap_res
    h = height / nb_pixels_y + snap_res

    ax = margin_x
    ay = margin_y

    xs = np.linspace(west + w / 2 - ax, east - w / 2 + ax, nb_pixels_x)
    ys = np.linspace(south + h / 2 - ay, north - h / 2 + ay, nb_pixels_y)
    xs = _snap_to_resolution(xs, snap_res)
    ys = _snap_to_resolution(ys, snap_res)
    grid = it.product(xs, ys)

    pixels = []

    for x, y in grid:
        _w, _s, _e, _n = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        newpix = geometry.box(_w, _s, _e, _n)
        if shape.intersects(newpix):
            r = (_w, _s, _e, _n)

            pixels.append(r)
    return pixels


def rect_to_coords(r):
    w, s, e, n = r
    return [(w, s), (w, n), (e, n), (e, s)]


def pixelate(pts, N=100, margin=0.4, **kwargs):
    """Pixelate shape defined by points Return rectangles [Rect1, Rect2, ...] \
    ready to go in the quad tree."""
    pixels = _pixelate(pts, N=N, margin=margin, **kwargs)
    return [rect_to_coords(pixel) for pixel in pixels]


def gen_pixels_op_blocking(pts, snap_res=0.05, margin=1.0, min_pixel_size=0.4):
    op_block_pts = polygon_grow(pts, margin)
    return pixelate(
        op_block_pts, min_pixel_size=min_pixel_size, N=100, snap_res=snap_res
    )


def gen_op_blocking(pts, snap_res=0.05, margin=0.3):
    return polygon_grow(pts, margin)


if __name__ == "__main__":
    import numpy as np

    pts = [(x, x**2) for x in np.linspace(0, 1, 5)]
    c = pixelate(pts)
    print(c)

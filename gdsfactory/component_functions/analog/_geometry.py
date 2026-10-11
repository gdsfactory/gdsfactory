"""Shared geometry primitives for analog inductor and transformer cells."""

from __future__ import annotations

import math
from collections.abc import Callable

from numpy import floor

from gdsfactory import Component, ComponentReference

Poly = list[tuple[float, float]]


def _zip(xs: list[float], ys: list[float]) -> Poly:
    return list(zip(xs, ys, strict=True))


def _map_y(p: Poly, f: Callable[[float], float]) -> Poly:
    return [(x, f(y)) for (x, y) in p]


def _mirror_x(p: Poly) -> Poly:
    return [(-x, y) for (x, y) in p]


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _make_aspect_shift_y(
    d_out: float, aspect_ratio: float = 1.0
) -> Callable[[float], float]:
    """Stretch straight sides only (y=0 fixed). Identity when aspect_ratio == 1."""
    if aspect_ratio == 1:
        return lambda y: y
    ext = d_out * (aspect_ratio - 1)
    return lambda y: y + ext / 2 if y > 0 else (y - ext / 2 if y < 0 else y)


def _routing_geometric_45(
    w: float, s: float, x0: float, y0: float, extend: float = 0.0
) -> Poly:
    """45-degree crossing routing polygon."""
    SQRT2 = math.sqrt(2)
    g = (SQRT2 - 1) * s
    d = (SQRT2 - 1) * w
    h = w + s + (SQRT2 - 1) * (2 * s + w)
    x_upper = [-h / 2, -h / 2 + g, h / 2 - g - d, h / 2]
    y_upper = [-s / 2, -s / 2, s / 2 + w, s / 2 + w]
    x_lower = [-h / 2, -h / 2 + g + d, h / 2 - g, h / 2]
    y_lower = [-s / 2 - w, -s / 2 - w, s / 2, s / 2]
    if extend > 0:
        x_upper = [-h / 2 - extend] + x_upper + [h / 2 + extend]
        y_upper = [-s / 2] + y_upper + [s / 2 + w]
        x_lower = [-h / 2 - extend] + x_lower + [h / 2 + extend]
        y_lower = [-s / 2 - w] + y_lower + [s / 2]
    xs = [v + x0 for v in x_upper] + [v + x0 for v in reversed(x_lower)]
    ys = [v + y0 for v in y_upper] + [v + y0 for v in reversed(y_lower)]
    return list(zip(xs, ys, strict=True))


def _via_component_info(
    via_component: Component,
) -> tuple[float, float, float, float, float]:
    """Validate and unpack the via geometry info gdsfactory via() cells carry.

    Returns (xsize, ysize, enclosure, column_pitch, row_pitch).
    """
    for key in ("xsize", "ysize", "enclosure", "column_pitch", "row_pitch"):
        if key not in via_component.info:
            raise ValueError(f"via {via_component.name!r} is missing {key!r} info")
    return (
        via_component.info["xsize"],
        via_component.info["ysize"],
        via_component.info["enclosure"],
        via_component.info["column_pitch"],
        via_component.info["row_pitch"],
    )


def _add_via_array(
    c: Component,
    via_component: Component,
    cx: float,
    cy: float,
    avail_x: float,
    avail_y: float,
) -> ComponentReference:
    """Place a via component array filling an avail_x x avail_y box centered at (cx, cy).

    avail_x/avail_y are the enclosure-netted region —
    same fill convention used across all four analog cells.
    """
    w, h, _enclosure, pitch_x, pitch_y = _via_component_info(via_component)

    nb_vias_x = int(floor((avail_x - w) / pitch_x + 1)) or 1
    nb_vias_y = int(floor((avail_y - h) / pitch_y + 1)) or 1
    nb_vias_x = max(nb_vias_x, 1)
    nb_vias_y = max(nb_vias_y, 1)

    via_ref = c.add_ref(
        via_component,
        columns=nb_vias_x,
        rows=nb_vias_y,
        column_pitch=pitch_x,
        row_pitch=pitch_y,
    )
    via_ref.move(
        (
            cx - via_ref.xsize / 2 + w / 2,
            cy - via_ref.ysize / 2 + h / 2,
        )
    )
    return via_ref


def _via_array_at(
    c: Component,
    via_component: Component,
    cx: float,
    cy: float,
    extend: float,
    width: float,
    enclosure: float,
) -> None:
    """Place a via array near (cx, cy), choosing the long axis (extend) to align with whichever of x/y is farther from the origin.

    Used for the top/bottom/left-right crossing vias, which need to sit under crossings routed in
    either the x or y direction depending on which quadrant they're in.
    """
    dx = _sign(cx) * (extend - width) / 2
    dy = _sign(cy) * (extend - width) / 2
    avail_extend = extend - 2 * enclosure
    avail_width = width - 2 * enclosure

    if abs(cy) > abs(cx):
        _add_via_array(c, via_component, cx + dx, cy, avail_extend, avail_width)
    else:
        _add_via_array(c, via_component, cx, cy + dy, avail_width, avail_extend)


def _pgs(D: float, w: float, s: float) -> list[Poly]:
    """Manhattan fishbone patterned ground shield filling a DxD square."""
    R = D / 2
    pitch = w + s
    sections: list[Poly] = []
    sections.append([(-w / 2, -R), (-w / 2, R), (w / 2, R), (w / 2, -R)])
    k_max = math.floor((R - w / 2) / pitch)
    for k in range(-k_max, k_max + 1):
        yc = k * pitch
        yb, yt = yc - w / 2, yc + w / 2
        sections.append([(-R, yb), (-R, yt), (R, yt), (R, yb)])
    return sections

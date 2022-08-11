"""Bends with grating couplers inside the spiral.

maybe: need to add grating coupler loopback as well
"""
from typing import Optional, Tuple

import numpy as np
from numpy import float64

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.types import ComponentSpec, CrossSectionSpec


def get_bend_port_distances(bend: Component) -> Tuple[float64, float64]:
    """Returns distance between bend ports."""
    p0, p1 = bend.ports.values()
    return abs(p0.x - p1.x), abs(p0.y - p1.y)


@cell
def spiral_external_io(
    N: int = 6,
    x_inner_length_cutback: float = 300.0,
    x_inner_offset: float = 0.0,
    y_straight_inner_top: float = 0.0,
    xspacing: float = 3.0,
    yspacing: float = 3.0,
    bend: ComponentSpec = bend_euler,
    length: Optional[float] = None,
    cross_section: CrossSectionSpec = "strip",
    **kwargs
) -> Component:
    """Returns spiral with input and output ports outside the spiral.

    Args:
        N: number of loops.
        x_inner_length_cutback: x inner length.
        x_inner_offset: x inner offset.
        y_straight_inner_top: y straight inner top.
        xspacing: center to center x-spacing.
        yspacing: center to center y-spacing.
        bend: function.
        length: length in um, it is the approximates total length.
        cross_section: spec.
        kwargs: cross_section settings.
    """
    if length:
        x_inner_length_cutback = length / (4 * (N - 1))

    y_straight_inner_top += 5
    x_inner_length_cutback += x_inner_offset
    _bend180 = bend(angle=180, cross_section=cross_section, **kwargs)
    _bend90 = bend(angle=90, cross_section=cross_section, **kwargs)

    bend_radius = _bend90.info["radius"]
    rx, ry = get_bend_port_distances(_bend90)
    _, rx180 = get_bend_port_distances(_bend180)  # rx180, second arg since we rotate

    component = Component()
    inner_loop_spacing = 2 * bend_radius + 5.0

    # Create manhattan path going from west grating to westest port of bend 180
    x_inner_length = x_inner_length_cutback + 5.0 + xspacing

    y_inner_bend = y_straight_inner_top - bend_radius - 5.0
    x_inner_loop = x_inner_length - 5.0
    p1 = (x_inner_loop, y_inner_bend)
    p2 = (x_inner_loop + inner_loop_spacing, y_inner_bend)

    _pt = np.array(p1)
    pts_w = [_pt]

    for i in range(N):
        y1 = y_straight_inner_top + ry + (2 * i + 1) * yspacing
        x2 = inner_loop_spacing + 2 * rx + x_inner_length + (2 * i + 1) * xspacing
        y3 = -ry - (2 * i + 2) * yspacing
        x4 = -(2 * i + 1) * xspacing
        if i == N - 1:
            x4 = x4 - rx180 + xspacing

        _pt1 = np.array([_pt[0], y1])
        _pt2 = np.array([x2, _pt1[1]])
        _pt3 = np.array([_pt2[0], y3])
        _pt4 = np.array([x4, _pt3[1]])
        _pt5 = np.array([_pt4[0], 0])
        _pt = _pt5

        pts_w += [_pt1, _pt2, _pt3, _pt4, _pt5]

    pts_w = pts_w[:-2]

    # Create manhattan path going from east grating to eastest port of bend 180
    _pt = np.array(p2)
    pts_e = [_pt]

    for i in range(N):
        y1 = y_straight_inner_top + ry + (2 * i) * yspacing
        x2 = inner_loop_spacing + 2 * rx + x_inner_length + 2 * i * xspacing
        y3 = -ry - (2 * i + 1) * yspacing
        x4 = -2 * i * xspacing

        _pt1 = np.array([_pt[0], y1])
        _pt2 = np.array([x2, _pt1[1]])
        _pt3 = np.array([_pt2[0], y3])
        _pt4 = np.array([x4, _pt3[1]])
        _pt5 = np.array([_pt4[0], 0])
        _pt = _pt5

        pts_e += [_pt1, _pt2, _pt3, _pt4, _pt5]

    pts_e = pts_e[:-2]

    # Join the two bits of paths and extrude the spiral geometry
    route = round_corners(
        pts_w[::-1] + pts_e,
        bend=bend,
        cross_section=cross_section,
        **kwargs,
    )

    component.add(route.references)
    component.add_port("o2", port=route.ports[0])
    component.add_port("o1", port=route.ports[1])

    length = route.length
    component.info["length"] = length
    return component


if __name__ == "__main__":
    c = spiral_external_io(auto_widen=True, width_wide=2.0, length=10e3, N=15)
    # print(c.info['length'])
    # print(c.info['length'] / 1e4, "cm")
    c.show(show_ports=True)

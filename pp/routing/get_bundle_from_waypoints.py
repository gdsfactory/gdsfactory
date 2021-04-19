from typing import Callable, List, Tuple, Union

import numpy as np
from numpy import float64, ndarray

from pp.components.bend_euler import bend_euler
from pp.components.straight import straight
from pp.components.taper import taper as taper_function
from pp.config import TAPER_LENGTH, WG_EXPANDED_WIDTH
from pp.port import Port
from pp.routing.manhattan import remove_flat_angles, round_corners
from pp.routing.utils import get_list_ports_angle
from pp.types import Coordinate, Coordinates, Number, Route


def _is_vertical(segment: Coordinate, tol: float = 1e-5) -> bool:
    p0, p1 = segment
    return abs(p0[0] - p1[0]) < tol


def _is_horizontal(segment: Coordinate, tol: float = 1e-5) -> bool:
    p0, p1 = segment
    return abs(p0[1] - p1[1]) < tol


def _segment_sign(s: Coordinate) -> Number:
    p0, p1 = s
    if _is_vertical(s):
        return np.sign(p1[1] - p0[1])

    if _is_horizontal(s):
        return np.sign(p1[0] - p0[0])


def get_ports_x_or_y_distances(
    list_ports: List[Port], ref_point: Coordinate
) -> List[Number]:
    if not list_ports:
        return []

    angle = get_list_ports_angle(list_ports)
    x0 = ref_point[0]
    y0 = ref_point[1]
    if angle in [0, 180]:
        xys = [p.y - y0 for p in list_ports]
        xs = [round(p.x, 5) for p in list_ports]
        if len(set(xs)) > 1:
            raise ValueError("List ports with angle 0/180 should all have the same x")
    else:
        xys = [p.x - x0 for p in list_ports]
        ys = [round(p.y, 5) for p in list_ports]
        if len(set(ys)) > 1:
            raise ValueError("List ports with angle 90/270 should all have the same y")
    return xys


def _distance(port1, port2):
    if hasattr(port1, "x"):
        x1, y1 = port1.x, port1.y
    else:
        x1, y1 = port1[0], port1[1]

    if hasattr(port2, "x"):
        x2, y2 = port2.x, port2.y
    else:
        x2, y2 = port2[0], port2[1]

    dx = x1 - x2
    dy = y1 - y2

    return np.sqrt(dx ** 2 + dy ** 2)


def get_bundle_from_waypoints(
    start_ports: List[Port],
    end_ports: List[Port],
    waypoints: Coordinates,
    straight_factory: Callable = straight,
    taper_factory: Callable = taper_function,
    bend_factory: Callable = bend_euler,
    bend_radius: float = 10.0,
    auto_sort: bool = True,
    **kwargs,
) -> List[Route]:
    """Returns list of routes that connect bundle of ports with bundle of routes
    where routes follow a list of waypoints.

    Args:
        start_ports: list of ports
        end_ports: list of ports
        waypoints: list of points defining a route
        straight_factory: function that returns straights
        taper_factory: function that returns tapers
        bend_factory: function that returns bends
        bend_radius: for bend
        auto_sort: sorts ports

    """
    if len(end_ports) != len(start_ports):
        raise ValueError(
            f"Number of start ports should match number of end ports.\
        Got {len(start_ports)} {len(end_ports)}"
        )

    for p in start_ports:
        p.angle = int(p.angle) % 360

    for p in end_ports:
        p.angle = int(p.angle) % 360

    start_angle = start_ports[0].orientation
    end_angle = end_ports[0].orientation

    # Sort the ports such that the bundle connect the correct corresponding ports.

    angles_to_sorttypes = {
        (0, 180): ("Y", "Y"),
        (0, 90): ("Y", "X"),
        (0, 0): ("Y", "-Y"),
        (0, 270): ("Y", "-X"),
        (90, 0): ("X", "Y"),
        (90, 90): ("X", "-X"),
        (90, 180): ("X", "-Y"),
        (90, 270): ("X", "X"),
        (180, 90): ("Y", "-X"),
        (180, 0): ("Y", "Y"),
        (180, 270): ("Y", "X"),
        (180, 180): ("Y", "-Y"),
        (270, 90): ("X", "X"),
        (270, 270): ("X", "-X"),
        (270, 0): ("X", "-Y"),
        (270, 180): ("X", "Y"),
    }

    dict_sorts = {
        "X": lambda p: p.x,
        "Y": lambda p: p.y,
        "-X": lambda p: -p.x,
        "-Y": lambda p: -p.y,
    }
    key = (start_angle, end_angle)
    sp_st, ep_st = angles_to_sorttypes[key]
    start_port_sort = dict_sorts[sp_st]
    end_port_sort = dict_sorts[ep_st]

    if auto_sort:
        start_ports.sort(key=start_port_sort)
        end_ports.sort(key=end_port_sort)

    routes = _generate_manhattan_bundle_waypoints(
        start_ports, end_ports, waypoints, **kwargs
    )

    bends90 = [bend_factory(radius=bend_radius, width=p.width) for p in start_ports]

    if taper_factory:
        if callable(taper_factory):
            taper = taper_factory(
                length=TAPER_LENGTH,
                width1=start_ports[0].width,
                width2=WG_EXPANDED_WIDTH,
                layer=start_ports[0].layer,
            )
        else:
            # In this case the taper is a fixed cell
            taper = taper_factory
    else:
        taper = None
    connections = [
        round_corners(
            points=pts,
            bend_factory=bend90,
            straight_factory=straight_factory,
            taper=taper,
        )
        for pts, bend90 in zip(routes, bends90)
    ]
    return connections


def snap_route_to_end_point_x(route, x):
    y1, y2 = [p[1] for p in route[-2:]]
    return route[:-2] + [(x, y1), (x, y2)]


def snap_route_to_end_point_y(
    route: List[Union[ndarray, Tuple[float64, float64]]], y: float64
) -> List[Union[ndarray, Tuple[float64, float64]]]:
    x1, x2 = [p[0] for p in route[-2:]]
    return route[:-2] + [(x1, y), (x2, y)]


def _generate_manhattan_bundle_waypoints(
    start_ports: List[Port],
    end_ports: List[Port],
    backbone_route: Coordinates,
    **kwargs,
) -> Coordinates:
    """
    Args:
        start_ports: list of start ports. Should all be facing in the same direction
        end_ports: list of end ports. Should all be facing in the same direction
        route: going from one point somewhere within the start_ports bank to
        another point within the end_ports bank
    """
    backbone_route = remove_flat_angles(backbone_route)

    way_segments = [(p0, p1) for p0, p1 in zip(backbone_route[:-1], backbone_route[1:])]
    offsets_start = get_ports_x_or_y_distances(start_ports, backbone_route[0])

    start_angle = start_ports[0].orientation
    if start_angle in [90, 270]:
        offsets_start = [-_d for _d in offsets_start]

    end_angle = end_ports[0].orientation

    def _displace_segment_copy(s, a, sh=1, sv=1):
        sign_seg = _segment_sign(s)
        if _is_horizontal(s):
            dp = (0, sh * sign_seg * a)
        elif _is_vertical(s):
            dp = (sv * sign_seg * a, 0)
        else:
            raise ValueError("Segment should be manhattan, got {}".format(s))

        displaced_seg = [np.array(p) + dp for p in s]
        return displaced_seg

    def _displace_segment_copy_group1(s, a):
        return _displace_segment_copy(s, a, sh=1, sv=-1)

    def _intersection(s1, s2):
        if _is_horizontal(s1) and _is_vertical(s2):
            sh, sv = s1, s2
        elif _is_horizontal(s2) and _is_vertical(s1):
            sh, sv = s2, s1
        else:

            if _is_horizontal(s1):
                s1_dir = "h"
            elif _is_vertical(s1):
                s1_dir = "v"
            else:
                s1_dir = "u"

            if _is_horizontal(s2):
                s2_dir = "h"
            elif _is_vertical(s2):
                s2_dir = "v"
            else:
                s2_dir = "u"

            raise ValueError(
                "s1 / s2 should be h/v or v/h. Got \
            {} {} {} {}".format(
                    s1_dir, s2_dir, s1, s2
                )
            )
        return (sv[0][0], sh[0][1])

    N = len(start_ports)
    routes = []
    _make_segment = _displace_segment_copy_group1
    for i in range(N):
        prev_seg_sep = offsets_start[i]
        route = []

        for j, seg in enumerate(way_segments):
            seg_sep = prev_seg_sep
            d_seg = _make_segment(seg, seg_sep)

            if j == 0:
                start_point = d_seg[0]
            else:
                prev_seg = way_segments[j - 1]
                tmp_seg = _make_segment(prev_seg, prev_seg_sep)
                start_point = _intersection(d_seg, tmp_seg)

            route += [start_point]

            # If last point before the ports, adjust the separation to the end ports
            if j == len(way_segments) - 1:
                end_point = end_ports[i].position
                route += [end_point]

                if end_angle in [0, 180]:
                    route = snap_route_to_end_point_y(route, end_point[1])
                else:
                    route = snap_route_to_end_point_x(route, end_point[0])

        routes += [route]

    return routes

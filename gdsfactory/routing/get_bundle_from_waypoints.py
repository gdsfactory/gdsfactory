from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
from numpy import float64, ndarray

import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.components.via_corner import via_corner
from gdsfactory.components.wire import wire_corner
from gdsfactory.cross_section import strip
from gdsfactory.port import Port
from gdsfactory.routing.manhattan import (
    RouteError,
    get_route_error,
    remove_flat_angles,
    round_corners,
)
from gdsfactory.routing.path_length_matching import path_length_matched_points
from gdsfactory.routing.utils import get_list_ports_angle
from gdsfactory.routing.validation import validate_connections
from gdsfactory.typings import (
    ComponentSpec,
    Coordinate,
    Coordinates,
    CrossSectionSpec,
    MultiCrossSectionAngleSpec,
    Number,
    Route,
)


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
    list_ports: list[Port], ref_point: Coordinate
) -> list[Number]:
    if not list_ports:
        return []

    angle = get_list_ports_angle(list_ports)
    x0 = ref_point[0]
    y0 = ref_point[1]
    return (
        [p.y - y0 for p in list_ports]
        if angle in [0, 180]
        else [p.x - x0 for p in list_ports]
    )


def _distance(port1, port2):
    x1, y1 = (port1.x, port1.y) if hasattr(port1, "x") else (port1[0], port1[1])
    x2, y2 = (port2.x, port2.y) if hasattr(port2, "x") else (port2[0], port2[1])
    dx = x1 - x2
    dy = y1 - y2

    return np.sqrt(dx**2 + dy**2)


def get_bundle_from_waypoints(
    ports1: list[Port],
    ports2: list[Port],
    waypoints: Coordinates,
    straight: ComponentSpec = straight_function,
    taper: ComponentSpec = taper_function,
    bend: ComponentSpec = bend_euler,
    sort_ports: bool = True,
    cross_section: CrossSectionSpec | MultiCrossSectionAngleSpec = strip,
    separation: float | None = None,
    on_route_error: Callable = get_route_error,
    path_length_match_loops: int | None = None,
    path_length_match_extra_length: float = 0.0,
    path_length_match_modify_segment_i: int = -2,
    **kwargs,
) -> list[Route]:
    """Returns list of routes that connect bundle of ports with bundle of routes.

    Routes follow a list of waypoints.
    Take a look at get_bundle_from_steps for easier definition.

    Args:
        ports1: list of ports.
        ports2: list of ports.
        waypoints: list of points defining a route.
        straight: function that returns straights.
        taper: function that returns tapers.
        bend: function that returns bends.
        sort_ports: sorts ports.
        cross_section: cross_section.
        separation: center to center, defaults to ports1 separation.
        on_route_error: function to call for routing errors.
        path_length_match_loops: Integer number of loops to add to bundle
            for path length matching (won't try to match if None).
        path_length_match_extra_length: Extra length to add
            to path length matching loops (requires path_length_match_loops != None).
        path_length_match_modify_segment_i: Index of straight segment to add path
            length matching loops to (requires path_length_match_loops != None).
        kwargs: cross_section settings.

    """
    _p1, _p2 = ports1, ports2
    if len(ports2) != len(ports1):
        raise ValueError(
            f"Number of start ports should match number of end ports.\
        Got {len(ports1)} {len(ports2)}"
        )

    waypoints = [ports1[0].center] + list(waypoints) + [ports2[0].center]
    for p in ports1:
        # if ports1 orientation is None, guess it from the first two waypoints
        if p.orientation is None:
            if np.subtract(waypoints[1], waypoints[0])[0] > 0:
                p.orientation = 0
            elif np.subtract(waypoints[1], waypoints[0])[0] < 0:
                p.orientation = 180
            elif np.subtract(waypoints[1], waypoints[0])[1] > 0:
                p.orientation = 90
            elif np.subtract(waypoints[1], waypoints[0])[1] < 0:
                p.orientation = 270
        p.orientation = int(p.orientation) % 360 if p.orientation else p.orientation

    for p in ports2:
        # if ports2 orientation is None, guess it from the last two waypoints
        if p.orientation is None:
            if np.subtract(waypoints[-1], waypoints[-2])[0] > 0:
                p.orientation = 180
            elif np.subtract(waypoints[-1], waypoints[-2])[0] < 0:
                p.orientation = 0
            elif np.subtract(waypoints[-1], waypoints[-2])[1] > 0:
                p.orientation = 90
            elif np.subtract(waypoints[-1], waypoints[-2])[1] < 0:
                p.orientation = 270
        p.orientation = int(p.orientation) % 360 if p.orientation else p.orientation

    if sort_ports:
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
        start_angle = ports1[0].orientation
        end_angle = ports2[0].orientation

        key = (start_angle, end_angle)
        sp_st, ep_st = angles_to_sorttypes[key]
        start_port_sort = dict_sorts[sp_st]
        end_port_sort = dict_sorts[ep_st]
        ports1.sort(key=start_port_sort)
        ports2.sort(key=end_port_sort)

    try:
        routes = _generate_manhattan_bundle_waypoints(
            ports1=ports1,
            ports2=ports2,
            waypoints=list(waypoints),
            separation=separation,
            **kwargs,
        )
    except RouteError:
        return [on_route_error(waypoints)]

    # bends90 = [
    #     gf.get_component(bend, cross_section=cross_section, **kwargs) for p in ports1
    # ]

    if taper and not isinstance(cross_section, list):
        x = gf.get_cross_section(cross_section, **kwargs)
        if x.auto_widen and callable(taper):
            taper = gf.get_component(
                taper,
                length=x.taper_length,
                width1=ports1[0].width,
                width2=x.width_wide,
                layer=ports1[0].layer,
            )
    else:
        taper = None

    if path_length_match_loops:
        routes = [np.array(route) for route in routes]
        routes = path_length_matched_points(
            routes,
            extra_length=path_length_match_extra_length,
            bend=bend,
            nb_loops=path_length_match_loops,
            modify_segment_i=path_length_match_modify_segment_i,
            cross_section=cross_section,
            **kwargs,
        )
    routes = [
        round_corners(
            points=pts,
            bend=bend,
            straight=straight,
            taper=taper,
            cross_section=cross_section,
            **kwargs,
        )
        for pts in routes
    ]
    validate_connections(_p1, _p2, routes)
    return routes


get_bundle_from_waypoints_electrical = partial(
    get_bundle_from_waypoints, bend=wire_corner, cross_section="metal_routing"
)

get_bundle_from_waypoints_electrical_multilayer = partial(
    get_bundle_from_waypoints_electrical,
    bend=via_corner,
    cross_section=[
        (gf.cross_section.metal2, (90, 270)),
        ("metal_routing", (0, 180)),
    ],
)


def snap_route_to_end_point_x(route, x):
    y1, y2 = (p[1] for p in route[-2:])
    return route[:-2] + [(x, y1), (x, y2)]


def snap_route_to_end_point_y(
    route: list[ndarray | tuple[float64, float64]], y: float64
) -> list[ndarray | tuple[float64, float64]]:
    x1, x2 = (p[0] for p in route[-2:])
    return route[:-2] + [(x1, y), (x2, y)]


def _generate_manhattan_bundle_waypoints(
    ports1: list[Port],
    ports2: list[Port],
    waypoints: Coordinates,
    separation: float | None = None,
    **kwargs,
) -> Coordinates:
    """Returns waypoints for bundle.

    Args:
        ports1: list of ports must face the same direction.
        ports2: list of ports must face the same direction.
        waypoints: from one point within the ports1 bank
            to another point within the ports2 bank.
        separation: center to center, defaults to ports1 separation.
    """
    waypoints = remove_flat_angles(waypoints)
    way_segments = list(zip(waypoints, waypoints[1:]))
    offsets_start = get_ports_x_or_y_distances(ports1, waypoints[0])

    start_angle = ports1[0].orientation
    if start_angle in [90]:
        offsets_start = [-_d for _d in offsets_start]
    end_angle = ports2[0].orientation

    # if separation is defined, the offsets should increment from the reference port
    # in the same direction as the original offsets
    if separation and len(ports1) > 1:
        # the default case when we start with the reference port
        offsets_start_sign = (
            np.sign(offsets_start[1]) if np.sign(offsets_start[1]) != 0 else 1
        )
        offsets_mid = [
            offsets_start_sign * separation * i for i, o in enumerate(offsets_start)
        ]
        if offsets_start[0] == 0:
            # the default case, no action necessary
            pass
        elif offsets_start[-1] == 0:
            # if the reference port is last, reverse the whole list
            offsets_mid.reverse()
        else:
            raise ValueError("Expected offset = 0 at either start or end of route.")

    # if there is only one route, we should skip this step as it is irrelevant
    else:
        # separation defaults to ports1 separation
        offsets_mid = offsets_start

    def _displace_segment_copy(s, a, sh=1, sv=1):
        sign_seg = _segment_sign(s)
        if _is_horizontal(s):
            dp = (0, sh * sign_seg * a)
        elif _is_vertical(s):
            dp = (sv * sign_seg * a, 0)
        else:
            raise RouteError(f"Segment should be manhattan, got {s}")

        return [np.array(p) + dp for p in s]

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
                f"s1 / s2 should be h/v or v/h. Got {s1_dir} {s2_dir} {s1} {s2}"
            )
        return sv[0][0], sh[0][1]

    routes = []
    _make_segment = _displace_segment_copy_group1
    for i, start_port in enumerate(ports1):
        route = []
        prev_seg_sep = None

        for j, seg in enumerate(way_segments):
            if j == 0:
                start_point = start_port.center
                seg_sep = offsets_start[i]
            else:
                seg_sep = offsets_mid[i]
                d_seg = _make_segment(seg, seg_sep)
                prev_seg = way_segments[j - 1]
                tmp_seg = _make_segment(prev_seg, prev_seg_sep)
                start_point = _intersection(d_seg, tmp_seg)

            route += [start_point]

            # If last point before the ports, adjust the separation to the end ports
            if j == len(way_segments) - 1:
                end_point = ports2[i].center
                route += [end_point]

                if end_angle in [0, 180]:
                    route = snap_route_to_end_point_y(route, end_point[1])
                else:
                    route = snap_route_to_end_point_x(route, end_point[0])
            prev_seg_sep = seg_sep
        routes += [route]

    return routes

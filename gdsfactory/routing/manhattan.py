import uuid
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import gdspy
import numpy as np
import pytest
from numpy import bool_, ndarray

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.bend_s import bend_s
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.cross_section import strip
from gdsfactory.geometry.functions import angles_deg
from gdsfactory.port import Port, select_ports_list
from gdsfactory.snap import snap_to_grid
from gdsfactory.tech import LAYER
from gdsfactory.types import (
    ComponentFactory,
    ComponentOrFactory,
    Coordinate,
    Coordinates,
    CrossSection,
    CrossSectionFactory,
    Layer,
    Route,
)

TOLERANCE = 0.0001
DEG2RAD = np.pi / 180
RAD2DEG = 1 / DEG2RAD

O2D = {0: "East", 180: "West", 90: "North", 270: "South"}


class RouteWarning(UserWarning):
    pass


class RouteError(ValueError):
    pass


def sign(x: float) -> int:
    return -1 if x < 0 else 1


def _get_unique_port_facing(
    ports: Dict[str, Port],
    orientation: int = 0,
    layer: Tuple[int, int] = (1, 0),
) -> List[Port]:
    """Ensures there is only one port"""
    ports_selected = select_ports_list(
        ports=ports, orientation=orientation, layer=layer
    )

    if len(ports_selected) > 1:
        orientation = orientation % 360
        direction = O2D[orientation]
        for port in ports_selected:
            print(port)
        raise ValueError(
            f"_get_unique_port_facing: \n\
            should have only one port facing {direction}\n\
            Got {len(ports_selected)} {[port.name for port in ports_selected]}"
        )

    return ports_selected


def _get_bend_ports(
    bend: Component,
    orientation: int = 0,
    layer: Tuple[int, int] = (1, 0),
) -> List[Port]:
    """Returns West and North facing ports for bend.

    Any standard bend/corner has two ports: one facing west and one facing north
    Returns these two ports in this order.
    """

    ports = bend.ports

    p_w = _get_unique_port_facing(ports=ports, orientation=180, layer=layer)
    p_n = _get_unique_port_facing(ports=ports, orientation=90, layer=layer)

    return p_w + p_n


def _get_straight_ports(
    straight: Component,
    layer: Tuple[int, int] = (1, 0),
) -> List[Port]:
    """Return West and east facing ports for straight waveguide.

    Any standard straight wire/straight has two ports:
    one facing west and one facing east
    """
    ports = straight.ports

    p_w = _get_unique_port_facing(ports=ports, orientation=180, layer=layer)
    p_e = _get_unique_port_facing(ports=ports, orientation=0, layer=layer)

    return p_w + p_e


def gen_sref(
    structure: Component,
    rotation_angle: int,
    x_reflection: bool,
    port_name: str,
    position: Coordinate,
) -> ComponentReference:
    """Place reference of `port_name` of `structure` at `position`.

    Keep this convention, otherwise phidl port transform won't work
    - 1 Mirror
    - 2 Rotate
    - 3 Move
    """
    position = np.array(position)

    if port_name is None:
        port_position = np.array([0, 0])
    else:
        port_position = structure.ports[port_name].midpoint

    ref = gf.ComponentReference(component=structure, origin=(0, 0))

    if x_reflection:  # Vertical mirror: Reflection across x-axis
        y0 = port_position[1]
        ref.reflect(p1=(0, y0), p2=(1, y0))

    ref.rotate(rotation_angle, center=port_position)
    ref.move(port_position, position)
    return ref


def _is_vertical(p0: ndarray, p1: ndarray) -> bool_:
    return np.abs(p0[0] - p1[0]) < TOLERANCE


def _is_horizontal(p0: ndarray, p1: ndarray) -> bool_:
    return np.abs(p0[1] - p1[1]) < TOLERANCE


def get_straight_distance(p0: ndarray, p1: ndarray) -> float:
    if _is_vertical(p0, p1):
        return np.abs(p0[1] - p1[1])
    if _is_horizontal(p0, p1):
        return np.abs(p0[0] - p1[0])

    raise RouteError(f"Waveguide points {p0} {p1} are not manhattan")


def transform(
    points: ndarray,
    translation: ndarray = (0, 0),
    angle_deg: int = 0,
    x_reflection: bool = False,
) -> ndarray:
    """Transform points.

    Args:
        points (np.array of shape (N,2) ): points to be transformed
        translation (2d like array): translation vector
        angle_deg: rotation angle
        x_reflection (bool): if True, mirror the shape across the x axis  (y -> -y)
    """
    # Copy
    pts = points[:, :]

    # Translate
    pts = pts + np.array(translation)

    # Rotate
    c = np.cos(DEG2RAD * angle_deg)
    s = np.sin(DEG2RAD * angle_deg)
    rotation_matrix = np.array([[c, s], [-s, c]])
    pts = np.dot(pts, rotation_matrix)

    # Mirror
    if x_reflection:
        pts[:, 1] = -pts[:, 1]

    return pts


def reverse_transform(
    points: ndarray,
    translation: Coordinate = (0, 0),
    angle_deg: int = 0,
    x_reflection: bool = False,
) -> ndarray:
    """
    Args:
        points (np.array of shape (N,2) ): points to be transformed
        translation (2d like array): translation vector
        angle_deg: rotation angle
        x_reflection: if True, mirror the shape across the x axis  (y -> -y)
    """
    angle_deg = -angle_deg

    # Copy
    pts = points[:, :]

    # Mirror
    if x_reflection:
        pts[:, 1] = -pts[:, 1]

    # Rotate
    c = np.cos(DEG2RAD * angle_deg)
    s = np.sin(DEG2RAD * angle_deg)
    rotation_matrix = np.array([[c, s], [-s, c]])
    pts = np.dot(pts, rotation_matrix)

    # Translate
    pts = pts - np.array(translation)

    return pts


def _generate_route_manhattan_points(
    input_port: Port,
    output_port: Port,
    bs1: float,
    bs2: float,
    start_straight_length: float = 0.01,
    end_straight_length: float = 0.01,
    min_straight_length: float = 0.01,
) -> ndarray:
    """Return list of ports for the route.

    Args:
        input_port:
        output_port:
        bs1: bend size
        bs2: bend size
        start_straight_length:
        end_straight_length:
        min_straight_length:
    """

    threshold = TOLERANCE

    # transform I/O to the case where output is at (0, 0) pointing east (180)
    p_input = input_port.midpoint
    p_output = output_port.midpoint
    pts_io = np.stack([p_input, p_output], axis=0)
    angle = output_port.orientation

    bend_orientation = -angle + 180
    transform_params = (-p_output, bend_orientation, False)

    _pts_io = transform(pts_io, *transform_params)
    p = _pts_io[0, :]
    _p_output = _pts_io[1, :]

    a = int(input_port.orientation + bend_orientation) % 360
    s = start_straight_length
    count = 0
    points = [p]

    while True:
        count += 1
        if count > 40:
            raise AttributeError(
                "Too many iterations for in {} -> out {}".format(
                    input_port, output_port
                )
            )
        # not ready for final bend: go over possibilities
        sigp = np.sign(p[1])
        if not sigp:
            sigp = 1
        if a % 360 == 0:
            # same directions
            if abs(p[1]) < threshold and p[0] <= threshold:
                # Reach the output!
                points += [_p_output]
                break
            elif (
                p[0] + (bs1 + bs2 + end_straight_length + s) < threshold
                and abs(p[1]) - (bs1 + bs2 + min_straight_length) > -threshold
            ):
                # sufficient space for S-bend
                p = (-end_straight_length - bs2, p[1])
                a = -sigp * 90
            elif (
                p[0]
                + (2 * bs1 + 2 * bs2 + end_straight_length + s + min_straight_length)
                < threshold
            ):
                # sufficient distance to move aside
                p = (p[0] + s + bs1, p[1])
                a = -sigp * 90
            elif abs(p[1]) - (2 * bs1 + 2 * bs2 + 2 * min_straight_length) > -threshold:
                p = (p[0] + s + bs1, p[1])
                a = -sigp * 90
            else:
                p = (p[0] + s + bs1, p[1])
                a = sigp * 90

        elif a == 180:
            # opposite directions
            if abs(p[1]) - (bs1 + bs2 + min_straight_length) > -threshold:
                # far enough: U-turn
                p = (min(p[0] - s, -end_straight_length) - bs2, p[1])
                a = -sigp * 90
            else:
                # more complex turn
                p = (
                    min(
                        p[0] - s - bs1,
                        -end_straight_length - min_straight_length - 2 * bs1 - bs2,
                    ),
                    p[1],
                )
                a = -sigp * 90
        elif a % 180 == 90:
            siga = -np.sign((a % 360) - 180)
            if not siga:
                siga = 1

            if ((-p[1] * siga) - (s + bs2) > -threshold) and (
                -p[0] - (end_straight_length + bs2)
            ) > -threshold:
                # simple case: one right angle to the end
                p = (p[0], 0)
                a = 0
            elif (p[1] * siga) <= threshold and p[0] + (
                end_straight_length + bs1
            ) > -threshold:
                # go to the west, and then turn upward
                # this will sometimes result in too sharp bends, but there is no
                # avoiding this!

                _y = min(
                    max(min(min_straight_length, 0.5 * abs(p[1])), abs(p[1]) - s - bs1),
                    bs1 + bs2 + min_straight_length,
                )

                p = (p[0], sigp * _y)
                if count == 1:  # take care of the start_straight case
                    p = (p[0], -sigp * max(start_straight_length, _y))

                a = 180
            elif (
                -p[0] - (end_straight_length + 2 * bs1 + bs2 + min_straight_length)
                > -threshold
            ):
                # go sufficiently up, and then east
                p = (
                    p[0],
                    siga * max(p[1] * siga + s + bs1, bs1 + bs2 + min_straight_length),
                )
                a = 0

            elif -p[0] - (end_straight_length + bs2) > -threshold:
                # make vertical S-bend to get sufficient room for movement
                points += [(p[0], p[1] + siga * (bs2 + s))]
                p = (
                    min(
                        p[0] - bs1 + bs2 + min_straight_length,
                        -2 * bs1 - bs2 - end_straight_length - min_straight_length,
                    ),
                    p[1] + siga * (bs2 + s),
                )
                # `a` remains the same
            else:
                # no viable solution for this case. May result in crossed straights
                p = (p[0], p[1] + sigp * (s + bs1))
                a = 180
        points += [p]
        s = min_straight_length + bs1

    points = np.stack([np.array(_p) for _p in points], axis=0)
    points = reverse_transform(points, *transform_params)
    return points


def _get_bend_reference_parameters(
    p0: ndarray, p1: ndarray, p2: ndarray, bend_cell: Component, port_layer: Layer
) -> Tuple[ndarray, int, bool]:
    """Returns bend reference settings.

    Args:
        p0: starting port waypoints
        p1: middle port waypoints
        p2: end port points

    8 possible configurations
    First mirror, Then rotate

    Returns:

    .. code::

       p2        p2
        |         |
        | C     A |
       p1-<-p0->-p1      dp1 horizontal
        |         |      dp2 vertical
        | D     B |
       p2        p2


       p2-<-p1->-p2
            |
          F |  E         dp1 vertical
            p0           dp2 horizontal
          H |  G
            |
       p2-<-p1->-p2

    """

    # is_horizontal(dp1), s1, s2 : transform (rotation, vertical mirror)
    transforms_map = {
        (True, 1, 1): (0, False),  # A No transform
        (True, 1, -1): (0, True),  # B Vertical mirror
        (True, -1, 1): (180, True),  # C Horizontal mirror
        (True, -1, -1): (180, False),  # D R180
        (False, 1, 1): (90, True),  # E R90 + vertical mirror
        (False, 1, -1): (90, False),  # F R270
        (False, -1, 1): (270, False),  # G R270
        (False, -1, -1): (270, True),  # H R270 + vertical mirror
    }

    b1, b2 = [p.midpoint for p in _get_bend_ports(bend=bend_cell, layer=port_layer)]

    bsx = b2[0] - b1[0]
    bsy = b2[1] - b1[1]

    dp1 = p1 - p0
    dp2 = p2 - p1
    is_h_dp1 = np.abs(dp1[1]) < TOLERANCE

    if is_h_dp1:
        xd1 = dp1[0]
        yd2 = dp2[1]
        s1 = sign(xd1)
        s2 = sign(yd2)

        bend_origin = p1 - np.array([s1 * bsx, 0])

    else:
        yd1 = dp1[1]
        xd2 = dp2[0]
        s1 = int(sign(yd1))
        s2 = int(sign(xd2))

        bend_origin = p1 - np.array([0, s1 * bsy])

    t = transforms_map[(is_h_dp1, s1, s2)]

    return bend_origin, t[0], t[1]


def make_ref(component_factory: Callable) -> Callable:
    def _make_ref(*args, **kwargs):
        return component_factory(*args, **kwargs).ref()

    return _make_ref


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


def get_route_error(
    points,
    cross_section: Optional[CrossSection] = None,
    layer_path: Layer = LAYER.ERROR_PATH,
    layer_label: Layer = LAYER.TEXT,
    layer_marker: Layer = LAYER.ERROR_MARKER,
    references: Optional[List[ComponentReference]] = None,
) -> Route:
    width = cross_section.info["width"] if cross_section else 10
    warnings.warn(
        f"Route error for points {points}",
        RouteWarning,
    )

    c = Component(f"route_{uuid.uuid4()}"[:16])
    path = gdspy.FlexPath(
        points,
        width=width,
        gdsii_path=True,
        layer=layer_path[0],
        datatype=layer_path[1],
    )
    c.add(path)
    ref = ComponentReference(c)
    port1 = gf.Port(name="p1", midpoint=points[0], width=width)
    port2 = gf.Port(name="p2", midpoint=points[1], width=width)

    point_marker = gf.c.rectangle(
        size=(width * 2, width * 2), centered=True, layer=layer_marker
    )
    point_markers = [point_marker.ref(position=point) for point in points] + [ref]
    labels = [
        gf.Label(
            text=str(i), position=point, layer=layer_label[0], texttype=layer_label[1]
        )
        for i, point in enumerate(points)
    ]

    references = references or []
    references += point_markers
    return Route(references=references, ports=[port1, port2], length=-1, labels=labels)


def round_corners(
    points: Coordinates,
    straight: ComponentFactory = straight_function,
    bend: ComponentFactory = bend_euler,
    bend_s_factory: Optional[ComponentFactory] = bend_s,
    taper: Optional[ComponentFactory] = None,
    straight_fall_back_no_taper: Optional[ComponentFactory] = None,
    mirror_straight: bool = False,
    straight_ports: Optional[List[str]] = None,
    cross_section: CrossSectionFactory = strip,
    on_route_error: Callable = get_route_error,
    with_point_markers: bool = False,
    snap_to_grid_nm: Optional[int] = 1,
    **kwargs,
) -> Route:
    """Returns Route:

    - references list with rounded straight route from a list of manhattan points.
    - ports: Tuple of ports
    - length: route length (float)

    Args:
        points: manhattan route defined by waypoints
        bend90: the bend to use for 90Deg turns
        straight: the straight library to use to generate straight portions
        taper: taper for straight portions. If None, no tapering
        straight_fall_back_no_taper: in case there is no space for two tapers
        mirror_straight: mirror_straight waveguide
        straight_ports: port names for straights. If None finds them automatically.
        cross_section:
        on_route_error: function to run when route fails
        with_point_markers: add route points markers (easy for debugging)
        snap_to_grid_nm: nm to snap to grid
        kwargs: cross_section settings
    """
    x = cross_section(**kwargs)
    points = (
        gf.snap.snap_to_grid(points, nm=snap_to_grid_nm) if snap_to_grid_nm else points
    )

    auto_widen = x.info.get("auto_widen", False)
    auto_widen_minimum_length = x.info.get("auto_widen_minimum_length", 200.0)
    taper_length = x.info.get("taper_length", 10.0)
    width = x.info.get("width", 2.0)
    width_wide = x.info.get("width_wide", None)
    references = []
    bend90 = bend(cross_section=cross_section, **kwargs) if callable(bend) else bend
    # bsx = bsy = _get_bend_size(bend90)
    taper = taper or taper_function(
        cross_section=cross_section,
        width1=width,
        width2=width_wide,
        length=taper_length,
    )
    taper = taper(cross_section=cross_section, **kwargs) if callable(taper) else taper

    # If there is a taper, make sure its length is known
    if taper and isinstance(taper, Component):
        if "length" not in taper.info:
            _taper_ports = list(taper.ports.values())
            taper.info["length"] = _taper_ports[-1].x - _taper_ports[0].x

    straight_fall_back_no_taper = straight_fall_back_no_taper or straight

    # Remove any flat angle, otherwise the algorithm won't work
    points = remove_flat_angles(points)
    points = np.array(points)

    straight_sections = []  # (p0, angle, length)
    p0_straight = points[0]
    p1 = points[1]

    total_length = 0  # Keep track of the total path length

    if not hasattr(bend90.info, "length"):
        raise ValueError(f"bend {bend90} needs to have bend.info.length defined")

    bend_length = bend90.info.length

    dp = p1 - p0_straight
    bend_orientation = None
    if _is_vertical(p0_straight, p1):
        if dp[1] > 0:
            bend_orientation = 90
        elif dp[1] < 0:
            bend_orientation = 270
    elif _is_horizontal(p0_straight, p1):
        if dp[0] > 0:
            bend_orientation = 0
        elif dp[0] < 0:
            bend_orientation = 180

    if bend_orientation is None:
        return on_route_error(points=points, cross_section=x)

    layer = x.info["layer"]
    try:
        pname_west, pname_north = [
            p.name for p in _get_bend_ports(bend=bend90, layer=layer)
        ]
    except ValueError as exc:
        raise ValueError(
            f"Did not find 2 ports on layer {layer}. Got {list(bend90.ports.values())}"
        ) from exc
    n_o_bends = points.shape[0] - 2
    total_length += n_o_bends * bend_length

    previous_port_point = points[0]
    bend_points = [previous_port_point]

    # Add bend sections and record straight-section information
    for i in range(1, points.shape[0] - 1):
        bend_origin, rotation, x_reflection = _get_bend_reference_parameters(
            points[i - 1], points[i], points[i + 1], bend90, x.info["layer"]
        )
        bend_ref = gen_sref(bend90, rotation, x_reflection, pname_west, bend_origin)
        references.append(bend_ref)

        dx_points = points[i][0] - points[i - 1][0]
        dy_points = points[i][1] - points[i - 1][1]

        if abs(dx_points) < TOLERANCE:
            matching_ports = [
                port
                for port in bend_ref.ports.values()
                if np.isclose(port.x, points[i][0])
            ]

        if abs(dy_points) < TOLERANCE:
            matching_ports = [
                port
                for port in bend_ref.ports.values()
                if np.isclose(port.y, points[i][1])
            ]

        if matching_ports:
            next_port = matching_ports[0]
            other_port_name = set(bend_ref.ports.keys()) - {next_port.name}
            other_port = bend_ref.ports[list(other_port_name)[0]]
            bend_points.append(next_port.midpoint)
            bend_points.append(other_port.midpoint)
            previous_port_point = other_port.midpoint

        try:
            straight_sections += [
                (
                    p0_straight,
                    bend_orientation,
                    get_straight_distance(p0_straight, bend_origin),
                )
            ]
        except RouteError:
            on_route_error(
                points=(p0_straight, bend_origin),
                cross_section=x,
                references=references,
            )

        p0_straight = bend_ref.ports[pname_north].midpoint
        bend_orientation = bend_ref.ports[pname_north].orientation

    bend_points.append(points[-1])

    try:
        straight_sections += [
            (
                p0_straight,
                bend_orientation,
                get_straight_distance(p0_straight, points[-1]),
            )
        ]
    except RouteError:
        on_route_error(
            points=(p0_straight, points[-1]),
            cross_section=x,
            references=references,
        )

    # with_point_markers=True
    # print()
    # for i, point in enumerate(points):
    #     print(i, point)
    # print()
    # for i, point in enumerate(bend_points):
    #     print(i, point)

    # ensure bend connectivity
    for i, point in enumerate(points[:-1]):
        sx = np.sign(points[i + 1][0] - point[0])
        sy = np.sign(points[i + 1][1] - point[1])
        bsx = np.sign(bend_points[2 * i + 1][0] - bend_points[2 * i][0])
        bsy = np.sign(bend_points[2 * i + 1][1] - bend_points[2 * i][1])
        if bsx * sx == -1 or bsy * sy == -1:
            return on_route_error(points=points, cross_section=x, references=references)

    wg_refs = []
    for straight_origin, angle, length in straight_sections:
        with_taper = False
        # wg_width = list(bend90.ports.values())[0].width
        length = snap_to_grid(length)
        total_length += length

        if auto_widen and length > auto_widen_minimum_length and width_wide:
            # Taper starts where straight would have started
            with_taper = True
            length = length - 2 * taper_length
            taper_origin = straight_origin

            pname_west, pname_east = [
                p.name for p in _get_straight_ports(taper, layer=x.info["layer"])
            ]
            taper_ref = taper.ref(
                position=taper_origin, port_id=pname_west, rotation=angle
            )

            references.append(taper_ref)
            wg_refs += [taper_ref]

            # Update start straight position
            straight_origin = taper_ref.ports[pname_east].midpoint

            # Straight waveguide
            kwargs_wide = kwargs.copy()
            kwargs_wide.update(width=width_wide)
            cross_section_wide = gf.partial(cross_section, **kwargs_wide)
            wg = straight(length=length, cross_section=cross_section_wide)
        else:
            wg = straight_fall_back_no_taper(
                length=length, cross_section=cross_section, **kwargs
            )

        if straight_ports is None:
            straight_ports = [
                p.name for p in _get_straight_ports(wg, layer=x.info["layer"])
            ]
        pname_west, pname_east = straight_ports

        wg_ref = wg.ref()
        wg_ref.move(wg.ports[pname_west], (0, 0))
        if mirror_straight:
            wg_ref.reflect_v(list(wg_ref.ports.values())[0].name)

        wg_ref.rotate(angle)
        wg_ref.move(straight_origin)

        if length > 0:
            references.append(wg_ref)
            wg_refs += [wg_ref]

        port_index_out = 1
        if with_taper:
            # Second taper:
            # Origin at end of straight waveguide, starting from east side of taper

            taper_origin = wg_ref.ports[pname_east]
            pname_west, pname_east = [
                p.name for p in _get_straight_ports(taper, layer=x.info["layer"])
            ]

            taper_ref = taper.ref(
                position=taper_origin,
                port_id=pname_east,
                rotation=angle + 180,
                v_mirror=True,
            )
            # references += [
            #     gf.Label(
            #         text=f"a{angle}",
            #         position=taper_ref.center,
            #         layer=2,
            #         texttype=0,
            #     )
            # ]
            references.append(taper_ref)
            wg_refs += [taper_ref]
            port_index_out = 0

    if with_point_markers:
        route = get_route_error(points, cross_section=x)
        references += route.references

    port_input = list(wg_refs[0].ports.values())[0]
    port_output = list(wg_refs[-1].ports.values())[port_index_out]
    length = snap_to_grid(float(total_length))
    return Route(references=references, ports=(port_input, port_output), length=length)


def generate_manhattan_waypoints(
    input_port: Port,
    output_port: Port,
    straight: ComponentFactory = straight_function,
    start_straight_length: Optional[float] = None,
    end_straight_length: Optional[float] = None,
    min_straight_length: Optional[float] = None,
    bend: ComponentFactory = bend_euler,
    cross_section: CrossSectionFactory = strip,
    **kwargs,
) -> ndarray:
    """Return waypoints for a Manhattan route between two ports."""

    bend90 = bend(cross_section=cross_section, **kwargs) if callable(bend) else bend
    x = cross_section(**kwargs)
    start_straight_length = start_straight_length or x.info.get("min_length")
    end_straight_length = end_straight_length or x.info.get("min_length")
    min_straight_length = min_straight_length or x.info.get("min_length")

    bsx = bsy = _get_bend_size(bend90)
    points = _generate_route_manhattan_points(
        input_port,
        output_port,
        bsx,
        bsy,
        start_straight_length,
        end_straight_length,
        min_straight_length,
    )
    return points


def _get_bend_size(bend90: Component):
    p1, p2 = list(bend90.ports.values())[:2]
    bsx = abs(p2.x - p1.x)
    bsy = abs(p2.y - p1.y)
    return max(bsx, bsy)


def route_manhattan(
    input_port: Port,
    output_port: Port,
    straight: ComponentFactory = straight_function,
    taper: Optional[ComponentOrFactory] = None,
    start_straight_length: Optional[float] = None,
    end_straight_length: Optional[float] = None,
    min_straight_length: Optional[float] = None,
    bend: ComponentFactory = bend_euler,
    cross_section: CrossSectionFactory = strip,
    with_point_markers: bool = False,
    **kwargs,
) -> Route:
    """Generates the Manhattan waypoints for a route.
    Then creates the straight, taper and bend references that define the route.
    """
    x = cross_section(**kwargs)

    start_straight_length = start_straight_length or x.info.get("min_length")
    end_straight_length = end_straight_length or x.info.get("min_length")
    min_straight_length = min_straight_length or x.info.get("min_length")

    points = generate_manhattan_waypoints(
        input_port,
        output_port,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        min_straight_length=min_straight_length,
        bend=bend,
        cross_section=cross_section,
        **kwargs,
    )
    return round_corners(
        points=points,
        straight=straight,
        taper=taper,
        bend=bend,
        cross_section=cross_section,
        with_point_markers=with_point_markers,
        **kwargs,
    )


def test_manhattan() -> Component:
    top_cell = Component()

    inputs = [
        Port("in1", (10, 5), 0.5, 90),
        # Port("in2", (-10, 20), 0.5, 0),
        # Port("in3", (10, 30), 0.5, 0),
        # Port("in4", (-10, -5), 0.5, 90),
        # Port("in5", (0, 0), 0.5, 0),
        # Port("in6", (0, 0), 0.5, 0),
    ]

    outputs = [
        Port("in1", (290, -60), 0.5, 180),
        # Port("in2", (-100, 20), 0.5, 0),
        # Port("in3", (100, -25), 0.5, 0),
        # Port("in4", (-150, -65), 0.5, 270),
        # Port("in5", (25, 3), 0.5, 180),
        # Port("in6", (0, 10), 0.5, 0),
    ]

    lengths = [349.974]

    for input_port, output_port, length in zip(inputs, outputs, lengths):

        # input_port = Port("input_port", (10,5), 0.5, 90)
        # output_port = Port("output_port", (90,-60), 0.5, 180)
        # bend = bend_circular(radius=5.0)

        route = route_manhattan(
            input_port=input_port,
            output_port=output_port,
            straight=straight_function,
            radius=5.0,
            auto_widen=True,
            width_wide=2,
            # layer=(2, 0),
            # width=0.2,
        )

        top_cell.add(route.references)
        assert np.isclose(route.length, length), route.length
    return top_cell


def test_manhattan_pass() -> Component:
    waypoints = [
        [10.0, 0.0],
        [20.0, 0.0],
        [20.0, 12.0],
        [120.0, 12.0],
        [120.0, 80.0],
        [110.0, 80.0],
    ]
    route = round_corners(waypoints, radius=5)
    c = Component()
    c.add(route.references)
    return c


def test_manhattan_fail() -> Component:
    waypoints = [
        [10.0, 0.0],
        [20.0, 0.0],
        [20.0, 12.0],
        [120.0, 12.0],
        [120.0, 80.0],
        [110.0, 80.0],
    ]
    with pytest.warns(RouteWarning):
        route = round_corners(waypoints, radius=10.0, with_point_markers=False)
    c = Component()
    c.add(route.references)
    return c


def _demo_manhattan_fail() -> Component:
    waypoints = [
        [10.0, 0.0],
        [20.0, 0.0],
        [20.0, 12.0],
        [120.0, 12.0],
        [120.0, 80.0],
        [110.0, 80.0],
    ]
    route = round_corners(waypoints, radius=10.0, with_point_markers=False)
    c = Component()
    c.add(route.references)
    return c


if __name__ == "__main__":
    # c = test_manhattan()
    # c = test_manhattan_fail()
    # c = test_manhattan_pass()
    # c = _demo_manhattan_fail()
    # c = gf.c.straight()
    # c = gf.routing.add_fiber_array(c)
    # c = gf.c.delay_snake()
    # c.show()

    c = gf.Component("pads_route_from_steps")
    pt = c << gf.c.pad_array(orientation=270, columns=3)
    pb = c << gf.c.pad_array(orientation=90, columns=3)
    pt.move((100, 200))
    route = gf.routing.get_route_from_steps(
        pt.ports["e11"],
        pb.ports["e11"],
        steps=[
            {"y": 100},
        ],
        cross_section=gf.cross_section.metal3,
        bend=gf.components.wire_corner,
    )
    c.add(route.references)
    c.show()

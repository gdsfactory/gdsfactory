from __future__ import annotations

import uuid
import warnings
from collections.abc import Callable
from functools import partial

import gdstk
import numpy as np
from numpy import bool_, ndarray

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.cross_section import strip
from gdsfactory.geometry.functions import angles_deg
from gdsfactory.port import Port, select_ports_list
from gdsfactory.routing.get_route_sbend import get_route_sbend
from gdsfactory.typings import (
    ComponentSpec,
    Coordinate,
    Coordinates,
    CrossSection,
    CrossSectionSpec,
    LayerSpec,
    LayerSpecs,
    MultiCrossSectionAngleSpec,
    Route,
)

nm = 1e-3
TOLERANCE = 1 * nm
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
    ports: dict[str, Port],
    orientation: float = 0,
    layer: LayerSpec | LayerSpecs = (1, 0),
) -> list[Port]:
    """Ensures there is only one port."""
    ports_selected = []
    if isinstance(layer, list):
        for _layer in layer:
            ports_selected = select_ports_list(
                ports=ports, orientation=orientation, layer=gf.get_layer(_layer)
            )
            if ports_selected:
                break
    else:
        ports_selected = select_ports_list(
            ports=ports, orientation=orientation, layer=gf.get_layer(layer)
        )

    if len(ports_selected) > 1:
        orientation %= 360
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
    orientation: float = 0,
    layer: LayerSpec | LayerSpecs = (1, 0),
) -> list[Port]:
    """Returns West and North facing ports for bend.

    Any standard bend/corner has two ports: one facing west and one
    facing north Returns these two ports in this order.

    """
    ports = bend.ports

    p_w = _get_unique_port_facing(ports=ports, orientation=180, layer=layer)
    p_n = _get_unique_port_facing(ports=ports, orientation=90, layer=layer)

    return p_w + p_n


def _get_straight_ports(
    straight: Component,
    layer: tuple[int, int] = (1, 0),
) -> list[Port]:
    """Return West and east facing ports for straight waveguide.

    Any standard straight wire/straight has two ports: one facing west
    and one facing east

    """
    ports = straight.ports

    p_w = _get_unique_port_facing(ports=ports, orientation=180, layer=layer)
    p_e = _get_unique_port_facing(ports=ports, orientation=0, layer=layer)

    return p_w + p_e


def gen_sref(
    structure: Component,
    rotation_angle: float,
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
        port_position = structure.ports[port_name].center

    ref = gf.ComponentReference(component=structure, origin=(0, 0))

    if x_reflection:  # Vertical mirror: Reflection across x-axis
        y0 = port_position[1]
        ref.mirror(p1=(0, y0), p2=(1, y0))

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
    print(f"Waveguide points {p0} {p1} are not manhattan")
    raise RouteError(f"Waveguide points {p0} {p1} are not manhattan")


def transform(
    points: ndarray,
    translation: ndarray,
    angle_deg: int = 0,
    x_reflection: bool = False,
) -> ndarray:
    """Transform points.

    Args:
        points (np.array of shape (N,2) ): points to be transformed.
        translation (2d like array): translation vector.
        angle_deg: rotation angle.
        x_reflection (bool): if True, mirror the shape across the x axis  (y -> -y).

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
    """Args are the following.

    points (np.array of shape (N,2) ): points to be transformed.
    translation (2d like array): translation vector.
    angle_deg: rotation angle.
    x_reflection: if True, mirror the shape across the x axis  (y -> -y).
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
        start_straight_length: in um.
        end_straight_length: in um.
        min_straight_length: in um.

    """
    threshold = TOLERANCE

    # transform I/O to the case where output is at (0, 0) pointing east (180)
    p_input = np.array(input_port.center)
    p_output = np.array(output_port.center)
    pts_io = np.stack([p_input, p_output], axis=0)
    angle = output_port.orientation

    if output_port.orientation is None and input_port.orientation is None:
        x0, y0 = p_input
        x2, y2 = p_output
        p1 = (x0, y2)
        points = np.array([p_input, p1, p_output])
    elif input_port.orientation is None:
        raise ValueError("input_port orientation is None")

    elif output_port.orientation is None:
        raise ValueError("output_port orientation is None")

    else:
        bend_orientation = -angle + 180
        transform_params = dict(
            translation=-p_output, angle_deg=bend_orientation, x_reflection=False
        )

        _pts_io = transform(pts_io, **transform_params)
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
                    f"Too many iterations for in {input_port} -> out {output_port}"
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
                    + (
                        2 * bs1
                        + 2 * bs2
                        + end_straight_length
                        + s
                        + min_straight_length
                    )
                    < threshold
                ):
                    # sufficient distance to move aside
                    p = (p[0] + s + bs1, p[1])
                    a = -sigp * 90
                elif (
                    abs(p[1]) - (2 * bs1 + 2 * bs2 + 2 * min_straight_length)
                    > -threshold
                ):
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
                        max(
                            min(min_straight_length, 0.5 * abs(p[1])),
                            abs(p[1]) - s - bs1,
                        ),
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
                        siga
                        * max(p[1] * siga + s + bs1, bs1 + bs2 + min_straight_length),
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
        points = reverse_transform(points, **transform_params)
    return points


def _get_bend_reference_parameters(
    p0: ndarray,
    p1: ndarray,
    p2: ndarray,
    bend_cell: Component,
    port_layer: LayerSpec | list[LayerSpec],
) -> tuple[ndarray, int, bool]:
    """Returns bend reference settings.

    Args:
        p0: starting port waypoints.
        p1: middle port waypoints.
        p2: end port points.
        bend_cell: bend component.
        port_layer: for the port.

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

    b1, b2 = (p.center for p in _get_bend_ports(bend=bend_cell, layer=port_layer))

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
    cross_section: CrossSection | None = None,
    layer_path: LayerSpec = (208, 0),
    layer_label: LayerSpec = (66, 0),
    layer_marker: LayerSpec = (207, 0),
    references: list[ComponentReference] | None = None,
    with_sbend: bool = False,
) -> Route:
    """Returns route with error markers.

    Args:
        points: route waypoints.
        cross_section: Optional cross_section.
        layer_path: for the error.
        layer_label: for the labels.
        layer_marker: for point markers.
        references: optional list of references.
        with_sbend: if True raises Error so we can use it in try, except
            if False raises Warning.
    """
    layer_path = gf.get_layer(layer_path)
    layer_label = gf.get_layer(layer_label)
    layer_marker = gf.get_layer(layer_marker)

    width = cross_section.width if cross_section else 10

    if with_sbend:
        raise RouteError(f"route error for points {points}")
    warnings.warn(f"Route error for points {points}", RouteWarning)

    c = Component(f"route_{uuid.uuid4()}"[:16])
    path = gdstk.FlexPath(
        points,
        width=width,
        simple_path=True,
        layer=layer_path[0],
        datatype=layer_path[1],
    )
    c.add(path)
    ref = ComponentReference(c)
    port1 = gf.Port(
        name="p1", center=points[0], width=width, layer=layer_path, orientation=0
    )
    port2 = gf.Port(
        name="p2", center=points[1], width=width, layer=layer_path, orientation=0
    )

    point_marker = gf.components.rectangle(
        size=(width * 2, width * 2), centered=True, layer=layer_marker
    )
    point_markers = [point_marker.ref(position=point) for point in points] + [ref]
    labels = [
        gf.Label(
            text=str(i), origin=point, layer=layer_label[0], texttype=layer_label[1]
        )
        for i, point in enumerate(points)
    ]

    references = references or []
    references += point_markers
    return Route(references=references, ports=(port1, port2), length=-1, labels=labels)


def round_corners(
    points: Coordinates,
    straight: ComponentSpec = straight_function,
    bend: ComponentSpec = bend_euler,
    taper: ComponentSpec | None = None,
    straight_fall_back_no_taper: ComponentSpec | None = None,
    mirror_straight: bool = False,
    straight_ports: list[str] | None = None,
    cross_section: CrossSectionSpec | MultiCrossSectionAngleSpec = strip,
    on_route_error: Callable = get_route_error,
    with_point_markers: bool = False,
    with_sbend: bool = False,
    **kwargs,
) -> Route:
    """Returns Route.

    - reference list with rounded straight route from a list of manhattan points.
    - ports: Tuple of ports.
    - length: route length (float).

    Args:
        points: manhattan route defined by waypoints.
        bend90: the bend to use for 90Deg turns.
        straight: the straight library to use to generate straight portions.
        taper: taper for straight portions. If None, no tapering.
        straight_fall_back_no_taper: in case there is no space for two tapers.
        mirror_straight: mirror_straight waveguide.
        straight_ports: port names for straights. If None finds them automatically.
        cross_section: spec.
        on_route_error: function to run when route fails.
        with_point_markers: add route points markers (easy for debugging).
        with_sbend: add sbend in case there are routing errors.
        kwargs: cross_section settings.

    """
    from gdsfactory.pdk import get_layer

    multi_cross_section = isinstance(cross_section, list)
    if multi_cross_section:
        x = [gf.get_cross_section(xsection[0], **kwargs) for xsection in cross_section]
        layer = [_x.layer for _x in x]
    else:
        x = gf.get_cross_section(cross_section, **kwargs)
        layer = x.layer

    layer = get_layer(layer)
    references = []

    bend90 = (
        bend
        if isinstance(bend, Component)
        else gf.get_component(bend, cross_section=cross_section, **kwargs)
    )

    # bsx = bsy = _get_bend_size(bend90)
    auto_widen = [_x.auto_widen for _x in x] if isinstance(x, list) else x.auto_widen

    auto_widen_minimum_length = (
        [_x.auto_widen_minimum_length for _x in x]
        if isinstance(x, list)
        else x.auto_widen_minimum_length
    )

    taper_length = (
        [_x.taper_length for _x in x] if isinstance(x, list) else x.taper_length
    )

    width = [_x.width for _x in x] if isinstance(x, list) else x.width
    width_wide = [_x.width_wide for _x in x] if isinstance(x, list) else x.width_wide

    if isinstance(cross_section, list):
        taper = None
    elif taper is None:
        taper = taper_function(
            cross_section=cross_section,
            width1=width,
            width2=width_wide,
            length=taper_length,
        )
    elif not isinstance(taper, Component):
        taper = gf.get_component(taper, cross_section=cross_section, **kwargs)

    # If there is a taper, make sure its length is known
    if taper and isinstance(taper, Component) and "length" not in taper.info:
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

    if not bend90.info.get("length"):
        raise ValueError(f"bend {bend90} needs to have bend.info['length'] defined")

    bend_length = bend90.info["length"]

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
        print(f"bend_orientation is None {p0_straight} {p1}")
        return on_route_error(
            points=points,
            cross_section=None if multi_cross_section else x,
            with_sbend=with_sbend,
        )

    try:
        pname_west, pname_north = (
            p.name for p in _get_bend_ports(bend=bend90, layer=layer)
        )
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
            points[i - 1], points[i], points[i + 1], bend90, layer
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
            bend_points.extend((next_port.center, other_port.center))
            previous_port_point = other_port.center

        try:
            straight_sections += [
                (
                    p0_straight,
                    bend_orientation,
                    get_straight_distance(p0_straight, bend_origin),
                )
            ]
        except RouteError as e:
            print(e)
            on_route_error(
                points=(p0_straight, bend_origin),
                cross_section=None if multi_cross_section else x,
                references=references,
                with_sbend=with_sbend,
            )

        p0_straight = bend_ref.ports[pname_north].center
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
    except RouteError as e:
        print(e)
        on_route_error(
            points=(p0_straight, points[-1]),
            cross_section=None if multi_cross_section else x,
            references=references,
            with_sbend=with_sbend,
        )

    # ensure bend connectivity
    for i, point in enumerate(points[:-1]):
        sx = np.sign(np.round(points[i + 1][0] - point[0], 3))
        sy = np.sign(np.round(points[i + 1][1] - point[1], 3))
        bsx = np.sign(np.round(bend_points[2 * i + 1][0] - bend_points[2 * i][0], 3))
        bsy = np.sign(np.round(bend_points[2 * i + 1][1] - bend_points[2 * i][1], 3))
        if bsx * sx == -1 or bsy * sy == -1:
            print(f"No enough space for a route between {point} and {points[i+1]}")
            return on_route_error(
                points=points,
                cross_section=None if multi_cross_section else x,
                references=references,
                with_sbend=with_sbend,
            )

    wg_refs = []
    for straight_origin, angle, length in straight_sections:
        if isinstance(cross_section, list):
            for section, angles in cross_section:
                if angle in angles:
                    xsection = section
                    break
        else:
            xsection = cross_section
        x = gf.get_cross_section(xsection, **kwargs)

        with_taper = False
        # wg_width = list(bend90.ports.values())[0].width
        total_length += length

        if (
            isinstance(cross_section, list)
            or not auto_widen
            or length <= auto_widen_minimum_length
            or not width_wide
        ):
            wg = gf.get_component(
                straight_fall_back_no_taper,
                length=length,
                cross_section=xsection,
                **kwargs,
            )
        else:
            # Taper starts where straight would have started
            with_taper = True
            length = length - 2 * taper_length
            taper_origin = straight_origin

            pname_west, pname_east = (
                p.name for p in _get_straight_ports(taper, layer=layer)
            )
            taper_ref = taper.ref(
                position=taper_origin, port_id=pname_west, rotation=angle
            )

            references.append(taper_ref)
            wg_refs += [taper_ref]

            # Update start straight position
            straight_origin = taper_ref.ports[pname_east].center

            # Straight waveguide
            kwargs_wide = kwargs.copy()
            kwargs_wide.update(width=width_wide)

            if callable(cross_section):
                cross_section_wide = partial(cross_section, **kwargs_wide)
            else:
                cross_section_wide = x.copy(width=width_wide)
            wg = gf.get_component(
                straight, length=length, cross_section=cross_section_wide
            )
        if straight_ports is None:
            straight_ports = [p.name for p in _get_straight_ports(wg, layer=layer)]

        pname_west, pname_east = straight_ports

        wg_ref = wg.ref()
        wg_ref.move(wg.ports[pname_west], (0, 0))
        if mirror_straight:
            wg_ref.mirror_y(list(wg_ref.ports.values())[0].name)

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
            pname_west, pname_east = (
                p.name for p in _get_straight_ports(taper, layer=layer)
            )

            taper_ref = taper.ref(
                position=taper_origin,
                port_id=pname_east,
                rotation=angle + 180,
                v_mirror=True,
            )
            references.append(taper_ref)
            wg_refs += [taper_ref]
            port_index_out = 0

    labels = []

    if with_point_markers:
        route = get_route_error(
            points, cross_section=None if multi_cross_section else x
        )

        references += route.references
        labels += route.labels

    port_input = list(wg_refs[0].ports.values())[0]
    port_output = list(wg_refs[-1].ports.values())[port_index_out]
    length = float(np.round(total_length, 3))
    return Route(
        references=references,
        ports=(port_input, port_output),
        length=length,
        labels=labels,
    )


def generate_manhattan_waypoints(
    input_port: Port,
    output_port: Port,
    start_straight_length: float | None = None,
    end_straight_length: float | None = None,
    min_straight_length: float | None = None,
    bend: ComponentSpec = bend_euler,
    cross_section: CrossSectionSpec | MultiCrossSectionAngleSpec = strip,
    **kwargs,
) -> ndarray:
    """Return waypoints for a Manhattan route between two ports.

    Args:
        input_port: source port.
        output_port: destination port.
        start_straight_length: Optional start length.
        end_straight_length: in um.
        min_straight_length: in um.
        bend: bend spec.
        cross_section: spec.
        kwargs: cross_section settings.

    """
    if "straight" in kwargs:
        _ = kwargs.pop("straight")

    bend90 = (
        bend
        if isinstance(bend, Component)
        else gf.get_component(bend, cross_section=cross_section, **kwargs)
    )

    if isinstance(cross_section, tuple | list):
        x = [gf.get_cross_section(xsection[0], **kwargs) for xsection in cross_section]
        start_straight_length = start_straight_length or min(_x.min_length for _x in x)
        end_straight_length = end_straight_length or min(_x.min_length for _x in x)
        min_straight_length = min_straight_length or min(_x.min_length for _x in x)
    else:
        x = gf.get_cross_section(cross_section, **kwargs)
        start_straight_length = start_straight_length or x.min_length
        end_straight_length = end_straight_length or x.min_length
        min_straight_length = min_straight_length or x.min_length

    bsx = bsy = _get_bend_size(bend90)
    return _generate_route_manhattan_points(
        input_port,
        output_port,
        bsx,
        bsy,
        start_straight_length,
        end_straight_length,
        min_straight_length,
    )


def _get_bend_size(bend90: Component):
    p1, p2 = list(bend90.ports.values())[:2]
    bsx = abs(p2.x - p1.x)
    bsy = abs(p2.y - p1.y)
    return max(bsx, bsy)


def route_manhattan(
    input_port: Port,
    output_port: Port,
    straight: ComponentSpec = straight_function,
    taper: ComponentSpec | None = None,
    start_straight_length: float | None = None,
    end_straight_length: float | None = None,
    min_straight_length: float | None = None,
    bend: ComponentSpec = bend_euler,
    with_sbend: bool = True,
    cross_section: CrossSectionSpec | MultiCrossSectionAngleSpec = strip,
    with_point_markers: bool = False,
    on_route_error: Callable = get_route_error,
    **kwargs,
) -> Route:
    """Generates the Manhattan waypoints for a route.

    Then creates the straight, taper and bend references that define the
    route, or create an SBend route.

    Args:
        input_port: input.
        output_port: output.
        straight: function.
        taper: add taper.
        start_straight_length: in um.
        end_straight_length: in um.
        min_straight_length: min length of straight for any intermediate segment.
        bend: bend spec.
        with_sbend: add sbend in case there are routing errors.
        cross_section: spec.
        with_point_markers: add point markers in the route.
        kwargs: cross_section settings.

    """
    if isinstance(cross_section, tuple | list):
        x = [gf.get_cross_section(xsection[0], **kwargs) for xsection in cross_section]
        start_straight_length = start_straight_length or min(_x.min_length for _x in x)
        end_straight_length = end_straight_length or min(_x.min_length for _x in x)
        min_straight_length = min_straight_length or min(_x.min_length for _x in x)
        x = cross_section
    else:
        x = gf.get_cross_section(cross_section, **kwargs)
        start_straight_length = start_straight_length or x.min_length
        end_straight_length = end_straight_length or x.min_length
        min_straight_length = min_straight_length or x.min_length

    try:
        points = generate_manhattan_waypoints(
            input_port,
            output_port,
            start_straight_length=start_straight_length,
            end_straight_length=end_straight_length,
            min_straight_length=min_straight_length,
            bend=bend,
            cross_section=x,
        )
        return round_corners(
            points=points,
            straight=straight,
            taper=taper,
            bend=bend,
            cross_section=x,
            with_point_markers=with_point_markers,
            with_sbend=with_sbend,
            on_route_error=on_route_error,
        )

    except RouteError:
        if with_sbend:
            return get_route_sbend(input_port, output_port, cross_section=x)

    return get_route_error(points=points, with_sbend=False)


if __name__ == "__main__":
    c = gf.Component("pads_route_from_steps")
    pt = c << gf.components.pad_array(orientation=270, columns=3)
    pb = c << gf.components.pad_array(orientation=90, columns=3)
    pt.move((100, 200))
    route = gf.routing.get_route_from_steps(
        pt.ports["e11"],
        pb.ports["e11"],
        steps=[
            {"y": 100},
        ],
        cross_section="metal_routing",
        bend=gf.components.wire_corner,
    )
    c.add(route.references)
    c.show(show_ports=True)

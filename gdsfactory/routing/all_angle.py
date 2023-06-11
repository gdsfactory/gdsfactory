import warnings
from typing import Callable, List, Optional

import numpy as np
import shapely.geometry as sg

from gdsfactory.component import Component, ComponentReference, Port
from gdsfactory.components.straight import straight
from gdsfactory.generic_tech.layer_map import LAYER
from gdsfactory.get_netlist import difference_between_angles
from gdsfactory.path import Path, extrude
from gdsfactory.routing.auto_taper import (
    _get_taper_io_port_names,
    taper_to_cross_section,
)
from gdsfactory.typings import STEP_DIRECTIVES_ALL_ANGLE as STEP_DIRECTIVES
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Route, StepAllAngle

BEND_PATH_FUNCS = {
    # 'euler_bend': euler_path,
}

Connector = Callable[..., List[ComponentReference]]


def get_connector(name: str) -> Connector:
    """
    Gets a connector function by name.

    Args:
        name: the name of the connector function to retrieve.

    Returns:
        The specified connector function.
    """
    try:
        connector = CONNECTORS[name]
    except KeyError as e:
        raise KeyError(
            f"{name} is not a valid connector type! Valid types are {list(CONNECTORS.keys())}"
        ) from e
    return connector


def vector_intersection(
    p0, a0, p1, a1, max_distance=100000, raise_error=True
) -> Optional[np.ndarray]:
    """
    Gets the intersection point between two vectors, specified by (point, angle) pairs, (p0, a0) and (p1, a1).

    Args:
        p0: x,y location of vector 0
        a0: angle of vector 0 [degrees]
        p1: x,y location of vector 1
        a1: angle of vector 1 [degrees]
        max_distance: maximum search distance for an intersection [um]
        raise_error: if True, raises an error if no intersection is found. Otherwise, returns None in that case.

    Returns:
        The (x,y) point of intersection, if one is found. Otherwise None.
    """
    a0_rad = np.deg2rad(a0)
    a1_rad = np.deg2rad(a1)
    dx0 = max_distance * np.cos(a0_rad)
    dy0 = max_distance * np.sin(a0_rad)
    p0_far = np.asarray(p0) + [dx0, dy0]
    l0 = sg.LineString([p0, p0_far])

    dx1 = max_distance * np.cos(a1_rad)
    dy1 = max_distance * np.sin(a1_rad)
    p1_far = np.asarray(p1) + [dx1, dy1]
    l1 = sg.LineString([p1, p1_far])

    intersect = l0.intersection(l1)
    if isinstance(intersect, sg.Point):
        return intersect.coords[0]
    if raise_error:
        raise ValueError(
            f"Vectors at {tuple(p0)} and {tuple(p1)} with angles {a0} and {a1} do not intersect!"
        )
    else:
        return None


def _line_intercept(p1, a1, p2, a2):
    if (((a2 - a1) % 180) + 180) % 180 == 0:
        raise ValueError("Lines are parallel!")

    k1 = np.tan(np.deg2rad(a1))
    k2 = np.tan(np.deg2rad(a2))
    x1, y1 = p1
    x2, y2 = p2
    if ((a1 % 180) + 180) % 180 == 90:
        return np.array((x1, k2 * (x1 - x2) + y2))
    elif ((a2 % 180) + 180) % 180 == 90:
        return np.array((x2, k1 * (x2 - x1) + y1))
    else:
        xi = (y1 - y2 - x1 * k1 + x2 * k2) / (k2 - k1)
        yi = k1 * (xi - x1) + y1

        return np.array((xi, yi))


def _get_bend_ports(bend):
    # this is a bit of a hack, but o1 < o2, in0 < out0, hopefully there are no other wacky conventions!
    sorted_port_names = sorted(bend.ports.keys())
    return [bend.ports[n] for n in sorted_port_names]


LOW_LOSS_CROSS_SECTIONS = [
    {"cross_section": "strip", "settings": {"width": 0.9}},
    "strip",
]


def low_loss_connector(
    port1: Port,
    port2: Port,
    prioritized_cross_sections: Optional[List[CrossSectionSpec]] = None,
    **kwargs,
) -> List[ComponentReference]:
    """
    Routes between two ports, using the lowest-loss cross-section which will fit.

    Args:
        port1: the starting port
        port2: the ending port
        prioritized_cross_sections: a list of cross-sections, sorted by preference (starting with most preferred). If None, uses the global variable LOW_LOSS_CROSS_SECTIONS

    Keyword Args:
        kwargs are added for API compatibility, but they are ignored.

    Returns:
        A list of component references comprising the connection
    """
    distance = np.sqrt(np.sum(np.square(port2.center - port1.center)))
    if prioritized_cross_sections is None:
        prioritized_cross_sections = LOW_LOSS_CROSS_SECTIONS
    # try to route with the lowest-loss cross-section
    for low_loss_cs in prioritized_cross_sections:
        taper1 = taper_to_cross_section(port1, cross_section=low_loss_cs)
        taper2 = taper_to_cross_section(port2, cross_section=low_loss_cs)
        taper_lengths = [
            taper.info["length"] for taper in (taper1, taper2) if taper is not None
        ]
        total_taper_length = sum(taper_lengths)
        if total_taper_length < distance:
            refs = []
            if taper1:
                output_port_name = _get_taper_io_port_names(taper1)[1]
                port1 = taper1.ports[output_port_name]
                refs.append(taper1)
            if taper2:
                output_port_name = _get_taper_io_port_names(taper2)[1]
                port2 = taper2.ports[output_port_name]
            intermediate_connector = straight_connector(
                port1, port2, cross_section=low_loss_cs
            )
            refs += intermediate_connector
            if taper2:
                refs.append(taper2)
            return refs
    if port1.cross_section == port2.cross_section:
        # if both cross-sections are the same, keep it
        return straight_connector(port1, port2, cross_section=port1.cross_section)
    elif port1.layer == port2.layer:
        # if the layer is the same, put a width taper, maximizing length of the fatty
        if port2.width > port1.width:
            taper = taper_to_cross_section(port1, port2.cross_section)
            refs = [taper]
            output_port_name = _get_taper_io_port_names(taper)[1]
            refs += straight_connector(
                taper.ports[output_port_name], port2, cross_section=port2.cross_section
            )
        else:
            taper = taper_to_cross_section(port2, port1.cross_section)
            output_port_name = _get_taper_io_port_names(taper1)[1]
            refs = straight_connector(
                port1, taper.ports[output_port_name], cross_section=port2.cross_section
            )
            refs.append(taper)
        return refs
    else:
        # if cross-sections are different, just put the cross-section at the start
        taper = taper_to_cross_section(port1, port2.cross_section)
        refs = [taper]
        output_port_name = _get_taper_io_port_names(taper1)[1]
        refs += straight_connector(
            taper.ports[output_port_name], port2, cross_section=port2.cross_section
        )
        return refs


def _make_error_trace(port1: Port, port2: Port, message: str):
    from gdsfactory.routing.manhattan import RouteWarning

    warnings.warn(message, RouteWarning)
    path = Path([port1.center, port2.center])
    error_component = extrude(path, layer=LAYER.ERROR_PATH, width=1)
    error_ref = ComponentReference(error_component)
    return [error_ref]


def straight_connector(
    port1: Port, port2: Port, cross_section: CrossSectionSpec = "strip"
) -> List[ComponentReference]:
    """
    Connects between the two ports with a straight of the given cross-section.

    Args:
        port1: the starting port.
        port2: the ending port.
        cross_section: the cross-section to use.

    Returns:
        A list of component references comprising the connection.
    """
    if np.array_equal(port1.center, port2.center):
        return []
    path = Path([port1.center, port2.center])
    # in usual cases, these angles should be the same, unless they are on opposite sides, in which they are 180 degrees separated
    if abs(difference_between_angles(path.start_angle, port1.orientation)) > 1:
        return _make_error_trace(
            port1,
            port2,
            message=f"Not enough room to route between ports: {port1} and {port2}",
        )

    length = np.linalg.norm(port1.center - port2.center)
    straight_component = straight(length=length, cross_section=cross_section)
    straight_ref = ComponentReference(straight_component)
    straight_ref.connect(list(straight_component.ports.keys())[0], port1)
    return [straight_ref]


def auto_taper_connector(
    port1: Port,
    port2: Port,
    cross_section: CrossSectionSpec = "strip",
    inner_connector: Connector = straight_connector,
) -> List[ComponentReference]:
    """
    Connects the two ports with a straight in the specified cross_section, adding tapers at either end if necessary.

    Args:
        port1: the first port.
        port2: the final port.
        cross_section: the primary cross section to use for the route.
        inner_connector: the connector to use after attaching tapers.

    Returns:
        A list of references comprising the connection.
    """
    taper1 = taper_to_cross_section(port1, cross_section)
    taper2 = taper_to_cross_section(port2, cross_section)
    route_refs = []
    if taper1:
        route_refs.append(taper1)
        output_port_name = _get_taper_io_port_names(taper1)[1]
        port1 = taper1.ports[output_port_name]
    if taper2:
        route_refs.append(taper2)
        output_port_name = _get_taper_io_port_names(taper2)[1]
        port2 = taper2.ports[output_port_name]
    conn = inner_connector(port1, port2, cross_section)
    route_refs += conn
    return route_refs


CONNECTORS = {
    "low_loss": low_loss_connector,
    "simple": straight_connector,
    "auto_taper": auto_taper_connector,
    None: straight_connector,
}
"""A dictionary of named connectors which can be used for all-angle routing"""


def _place_bend(bend_component: Component, position, rotation) -> ComponentReference:
    """
    Places a bend by its control point at a given position and rotation. The control point of a bend is the intersection of the inverted port vectors.

    Args:
        bend_component: the bend component
        position: the (x,y) position to place the bend
        rotation: the rotation of the bend
    Returns:
        The resulting bend ComponentReference
    """
    bend_ports = _get_bend_ports(bend_component)
    bend_control_point = vector_intersection(
        bend_ports[0].center,
        bend_ports[0].orientation + 180,
        bend_ports[1].center,
        bend_ports[1].orientation + 180,
    )
    bend_ref = ComponentReference(bend_component)
    bend_ref.rotate(
        rotation + 180 - bend_ports[0].orientation, center=bend_control_point
    )
    bend_ref.move(origin=bend_control_point, destination=position)
    return bend_ref


def _point_intersects_ray(p0, a0, p1, angle_tolerance=1e-4):
    x0, y0 = p0
    x1, y1 = p1
    a1 = np.arctan2(y1 - y0, x1 - x0)
    a1 = np.rad2deg(a1)
    return abs(difference_between_angles(a1, a0)) < angle_tolerance


def _null_handler(refs):
    return None


def _all_angle_connector(
    port1: Port,
    port2: Port,
    bend_angle: float,
    intersect: np.ndarray,
    bend: ComponentSpec = "euler_bend",
    cross_section: CrossSectionSpec = "strip",
    connector1: Connector = straight_connector,
    cross_section1: Optional[CrossSectionSpec] = None,
    connector2: Connector = straight_connector,
    cross_section2: Optional[CrossSectionSpec] = None,
    report_segment_separation: Optional[
        Callable[[List[ComponentReference]], None]
    ] = None,
):
    if cross_section1 is None:
        cross_section1 = cross_section
    if cross_section2 is None:
        cross_section2 = cross_section
    if report_segment_separation is None:
        report_segment_separation = _null_handler

    # in the case that the two ports already directly align
    if bend_angle == 0 and _point_intersects_ray(
        port1.center, port1.orientation, port2.center
    ):
        straight_connection = connector2(port1, port2, cross_section=cross_section2)
        report_segment_separation(straight_connection)
        return straight_connection
    if intersect is None:
        # if difference_between_angles(port2.orientation, port1.orientation) == 180:
        sample_bend = _get_bend(bend, angle=90, cross_section=cross_section)
        bend_cs = _get_bend_ports(sample_bend)[0].cross_section
        taper1 = taper_to_cross_section(port1, bend_cs)
        taper2 = taper_to_cross_section(port2, bend_cs)
        route_refs = []
        if taper1:
            route_refs.append(taper1)
            output_port_name = _get_taper_io_port_names(taper1)[1]
            port1 = taper1.ports[output_port_name]
        if taper2:
            route_refs.append(taper2)
            output_port_name = _get_taper_io_port_names(taper2)[1]
            port2 = taper2.ports[output_port_name]
        # try:
        bend_angles = _get_bend_angles(
            port1.center, port2.center, port1.orientation, port2.orientation, bend=bend
        )
        bends = [
            _get_bend(bend, angle=bend_angle, cross_section=cross_section)
            for bend_angle in bend_angles
        ]
        bend_refs = [ComponentReference(b) for b in bends]
        bend_refs_ports = [_get_bend_ports(br) for br in bend_refs]
        bend_refs[0].connect(bend_refs_ports[0][0], port1)
        bend_refs[1].connect(bend_refs_ports[1][0], port2)
        bend_refs_ports = [_get_bend_ports(br) for br in bend_refs]
        connection = connector2(
            bend_refs_ports[0][1], bend_refs_ports[1][1], cross_section=cross_section2
        )
        route_refs += bend_refs + connection
        report_segment_separation(connection + [bend_refs[0]])
        # except Exception as e:
        #     failure_message = f'Unable to complete route! Error message when attempting to create S bend between ports at {port1.center} and {port2.center}: {e}'
        #     route_refs += _make_error_trace(port1, port2, message=failure_message)
        return route_refs
        # return _make_error_trace(port1, port2, f'Port vectors do not intersect: {port1} and {port2}')
    bend_component = _get_bend(bend, angle=bend_angle, cross_section=cross_section)
    bend_ref = _place_bend(
        bend_component, position=intersect, rotation=port1.orientation
    )
    bend_ref_ports = _get_bend_ports(bend_ref)

    straight1 = connector1(port1, bend_ref_ports[0], cross_section=cross_section1)
    straight2 = connector2(bend_ref_ports[1], port2, cross_section=cross_section2)
    route_refs = straight1 + [bend_ref] + straight2
    report_segment_separation(straight1)
    return route_refs


def _get_bend(
    component: ComponentSpec,
    angle: float,
    cross_section: CrossSectionSpec,
    angle_precision: int = 9,
):
    from gdsfactory.pdk import get_component

    if (
        isinstance(component, dict)
        and "settings" in component
        and "cross_section" in component["settings"]
    ):
        return get_component(component, angle=round(angle, angle_precision))
    return get_component(
        component, angle=round(angle, angle_precision), cross_section=cross_section
    )


def _get_bend_angles(p0, p1, a0, a1, bend):
    """get the direct line between the two points."""
    import scipy.optimize

    from gdsfactory.pdk import get_component

    a_connect = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    a_connect_deg = np.rad2deg(a_connect)
    # these are the angles which should be swept by the bends, if the bends were to take up no space
    bend_angle_ideal_0 = difference_between_angles(a_connect_deg, a0)
    bend_angle_ideal_1 = difference_between_angles(a_connect_deg + 180, a1)
    # retrieves a function to calculate just the bend path (for efficiency) if available
    bend_path_func = BEND_PATH_FUNCS.get(bend)

    def optimization_func(d_angle):
        # apply a delta to both bend angles
        bend_angle0 = bend_angle_ideal_0 + d_angle
        bend_angle1 = bend_angle_ideal_1 + d_angle
        if bend_path_func is None:
            bend0 = get_component(bend, angle=bend_angle0).ref()
            bend1 = get_component(bend, angle=bend_angle1).ref()
            bend0 = bend0.rotate(a0).move(p0)
            bend1 = bend1.rotate(a1).move(p1)
            bend0_output_port = _get_bend_ports(bend0)[1]
            bend1_output_port = _get_bend_ports(bend1)[1]
            dx, dy = bend1_output_port.center - bend0_output_port.center
        else:
            bend0 = Path(bend_path_func(angle=bend_angle0))
            bend1 = Path(bend_path_func(angle=bend_angle1))
            # get rotated coordinates of bends and check if dx/dy match up to tan(angle)
            bend0 = bend0.rotate(a0).move(p0)
            bend1 = bend1.rotate(a1).move(p1)
            dx, dy = bend1.points[-1] - bend0.points[-1]
        angle_est = np.rad2deg(np.arctan2(dy, dx))
        angle_actual = a0 + bend_angle0
        angle_error = abs(difference_between_angles(angle_actual, angle_est))
        return angle_error

    result = scipy.optimize.minimize_scalar(optimization_func, bounds=(-45, 45))
    d_angle = result.x
    bend_angle_0 = bend_angle_ideal_0 + d_angle
    bend_angle_1 = bend_angle_ideal_1 + d_angle
    return bend_angle_0, bend_angle_1


def _get_minimum_separation(refs: List[ComponentReference], *ports) -> float:
    all_ports = [p for ref in refs for p in ref.ports.values()]
    all_ports.extend(ports)
    max_specified_separation = 0
    for port in all_ports:
        if port.cross_section:
            xs = port.cross_section
            separation = xs.gap + xs.width
            if separation > max_specified_separation:
                max_specified_separation = separation

    if max_specified_separation == 0:
        raise ValueError(
            "Cannot automatically determine separation. No ports in route have a cross_section which declares a default separation value!"
        )
    return max_specified_separation


def _points_approx_equal(
    point1: np.ndarray, point2: np.ndarray, tolerance: float = 5e-4
) -> bool:
    return np.sqrt(np.sum(np.square(point1 - point2))) < tolerance


def _angles_approx_opposing(angle1: float, angle2: float, tolerance: float = 1e-4):
    return abs(difference_between_angles(angle1 + 180, angle2)) < tolerance


def get_bundle_all_angle(
    ports1: List[Port],
    ports2: List[Port],
    steps: Optional[List[StepAllAngle]] = None,
    cross_section: CrossSectionSpec = "strip",
    bend: ComponentSpec = "bend_euler",
    connector: str = "low_loss",
    start_angle: Optional[float] = None,
    end_angle: Optional[float] = None,
    end_connector: Optional[str] = None,
    end_cross_section: Optional[CrossSectionSpec] = None,
    separation: Optional[float] = None,
    **kwargs,
) -> List[Route]:
    """Connects a bundle of ports, allowing steps which create waypoints at \
            arbitrary, non-manhattan angles.

    Args:
        ports1: ports at the start of the bundle.
        ports2: ports at the end of the bundle.
        steps: a list of steps, which contain directives on how to proceed with the route.
            "x", "y", "dx", "dy", "ds", "exit_angle", "cross_section", "connector", "separation".
            The first route, between ports1[0] and ports2[0] will take on the role of the primary route,
            and other routes will follow, given the bundling logic.
            It is assume that both ports1 and ports2 are sorted.
        cross_section: the default cross-section of the bends.
            Then the specified connector may also use this information for straights in between.
        bend: the default component to use for the bends.
        connector: the default connector to use to connect between two ports.
        start_angle: if defined and different from the angle of port1,
            will cap the starting port with a bend, as to exit with this angle.
        end_angle:  if defined, and different from the angle of port2,
            will cap the ending port with a bend, as to exit with this angle.
        end_connector: specifies the connector to use for the final straight segment of the route.
        end_cross_section: specifies the cross section to use for the final straight segment of the route.
        separation: specifies the separation between adjacent routes.
            If None, will query each segment's cross-section's and choose the largest value.
        kwargs: added for compatibility, but in general, kwargs will be ignored with a warning.

    Returns:
        List of Routes between ports1 and ports2.

    .. plot::
        :include-source:

        import gdsfactory as gf
        c = gf.Component("demo")

        mmi = gf.components.mmi2x2(width_mmi=10, gap_mmi=3)
        mmi1 = c << mmi
        mmi2 = c << mmi

        mmi2.move((100, 30))
        mmi2.rotate(30)

        routes = gf.routing.get_bundle_all_angle(
            mmi1.get_ports_list(orientation=0),
            [mmi2.ports["o2"], mmi2.ports["o1"]],
            connector=None,
        )
        for route in routes:
            c.add(route.references)
        c.plot()


    """
    from gdsfactory.pdk import get_cross_section

    if kwargs:
        warnings.warn(
            f"Unrecognized arguments for all-angle route will be ignored: {kwargs}"
        )

    connector_func = get_connector(connector)
    routes = []
    is_primary_route = True
    final_connector_func = connector_func
    final_cross_section = cross_section
    waypoints, angles = None, None

    # by default, the second to last segment (in case of a two-step end connection) should have default
    # connector and cross-section
    semi_final_connector_func = connector_func
    semi_final_cross_section = cross_section
    # however, this can be overridden by providing a final step without any directional arguments
    if steps and {"connector", "cross_section"}.issuperset(steps[-1]):
        final_step = steps[-1]
        steps = steps[:-1]
        if "cross_section" in final_step:
            semi_final_cross_section = final_step["cross_section"]
            semi_final_connector_func = auto_taper_connector
        if "connector" in final_step:
            semi_final_connector_func = get_connector(final_step["connector"])

    segment_separations = []

    if end_connector or end_cross_section:
        if end_cross_section:
            final_cross_section = end_cross_section
            final_connector_func = auto_taper_connector
        if end_connector:
            final_connector_func = get_connector(end_connector)

    for port1, port2 in zip(ports1, ports2):
        if _points_approx_equal(port1.center, port2.center) and _angles_approx_opposing(
            port1.orientation, port2.orientation
        ):
            continue
        route_refs = []
        if (
            start_angle is not None
            and difference_between_angles(start_angle, port1.orientation) != 0
        ):
            bend_angle = difference_between_angles(start_angle, port1.orientation)
            bend_component = _get_bend(
                bend, angle=bend_angle, cross_section=cross_section
            )
            bend_ref = ComponentReference(bend_component)
            bend_ref_ports = _get_bend_ports(bend_ref)
            initial_taper = taper_to_cross_section(
                port1, bend_ref_ports[0].cross_section
            )
            if initial_taper:
                route_refs.append(initial_taper)
                output_port_name = _get_taper_io_port_names(initial_taper)[1]
                port1 = initial_taper.ports[output_port_name]
            bend_ref.connect(bend_ref_ports[0], port1)
            bend_ref_ports = _get_bend_ports(bend_ref)
            route_refs.append(bend_ref)
            port1 = bend_ref_ports[1]
        if (
            end_angle is not None
            and difference_between_angles(end_angle, port2.orientation) != 0
        ):
            bend_angle = difference_between_angles(end_angle, port2.orientation)
            bend_component = _get_bend(
                bend, angle=bend_angle, cross_section=cross_section
            )
            bend_ref = ComponentReference(bend_component)
            bend_ref_ports = _get_bend_ports(bend_ref)
            end_taper = taper_to_cross_section(port2, bend_ref_ports[0].cross_section)
            if end_taper:
                route_refs.append(end_taper)
                output_port_name = _get_taper_io_port_names(end_taper)[1]
                port2 = end_taper.ports[output_port_name]
            bend_ref.connect(bend_ref_ports[0], port2)
            bend_ref_ports = _get_bend_ports(bend_ref)
            route_refs.append(bend_ref)
            port2 = bend_ref_ports[1]

        if not is_primary_route and steps:
            # for non-primary routes in the bundle, reset the steps for each new route,
            # based on the primary route's waypoints and angles
            these_waypoints = [port1.center]
            these_angles = [port1.orientation]
            i_step = 0
            intercept_sign = (
                1
                if vector_intersection(
                    these_waypoints[i_step],
                    these_angles[i_step],
                    waypoints[i_step + 1],
                    angles[i_step + 1] + 90,
                    raise_error=False,
                )
                is not None
                else -1
            )
            for i_waypoint in range(1, len(waypoints) - 2):
                # here we need the pitch for the *next* segment, after the bend
                pitch = segment_separations[i_waypoint]
                # the angle orthogonal from the next section's angle of propagation
                offset_angle = angles[i_waypoint] + 90 * intercept_sign

                # offset the next waypoint out by the desired pitch
                offset_pt = waypoints[i_waypoint] + pitch * np.array(
                    [np.cos(np.deg2rad(offset_angle)), np.sin(np.deg2rad(offset_angle))]
                )
                # the next waypoint will be the intersect of the current vector and the line offset from the previous route's next segment
                next_waypoint = _line_intercept(
                    offset_pt,
                    angles[i_waypoint],
                    these_waypoints[i_waypoint - 1],
                    these_angles[i_waypoint - 1],
                )

                these_waypoints.append(next_waypoint)
                these_angles.append(angles[i_waypoint])
            # final_intercept = vector_intersection(these_waypoints[-1], these_angles[-1], port2.center,
            #                                       port2.orientation, raise_error=True)
            waypoints = these_waypoints
            angles = these_angles
            has_explicit_end_angle = True

        if steps and is_primary_route:
            x0, y0 = port1.center
            a0 = port1.orientation
            waypoints = [(x0, y0)]
            angles = [port1.orientation]
            a_final = None

            # for each step, get the next waypoint
            last_step_index = len(steps) - 1
            for i_step, step in enumerate(steps):
                x1, y1 = None, None
                if not STEP_DIRECTIVES.issuperset(step):
                    invalid_step_directives = list(set(step.keys()) - STEP_DIRECTIVES)
                    raise ValueError(
                        f"Invalid step directives: {invalid_step_directives}. Valid directives are {list(STEP_DIRECTIVES)}"
                    )
                if not {"x", "y", "dx", "dy", "ds"}.isdisjoint(step):
                    if "x" in step or "dx" in step:
                        x1 = step.get("x", x0)
                        x1 += step.get("dx", 0)
                    if "y" in step or "dy" in step:
                        y1 = step.get("y", y0)
                        y1 += step.get("dy", 0)
                    if x1 is not None and y1 is not None and a0 is not None:
                        raise ValueError(
                            "Route is overconstrained! x and y and incoming angle are all defined. Please remove one"
                        )
                    if "ds" in step:
                        if {"x", "y", "dx", "dy"}.isdisjoint(step):
                            if a0 is not None:
                                ds = step["ds"]
                                dx = ds * np.cos(np.deg2rad(a0))
                                dy = ds * np.sin(np.deg2rad(a0))
                                x1 = x0 + dx
                                y1 = y0 + dy
                                if i_step == last_step_index:
                                    a_final = a0
                            else:
                                raise ValueError(
                                    'When specifying "ds" as a step, the previous step must have an explicit exit_angle'
                                )
                        else:
                            raise ValueError(
                                f"Route is overconstrained! ds is defined as well as x/y/dx/dy: {step}"
                            )
                    if x1 is None:
                        if a0 is not None:
                            # get intercept with desired y value
                            x1, _ = vector_intersection(
                                (x0, y0), a0, (-1e6, y1), 0, max_distance=2e6
                            )
                            if i_step == last_step_index:
                                a_final = a0
                        elif i_step == last_step_index:
                            if "exit_angle" in step:
                                # if exit_angle is set, assume the segment is vertical, from the previous point to the specified y
                                x1 = x0
                            else:
                                # otherwise, let x be at the intercept of the specified y with the ray defined by port 2's vector
                                x1, _ = vector_intersection(
                                    port2.center,
                                    port2.orientation,
                                    (-1e6, y1),
                                    0,
                                    max_distance=2e6,
                                )
                        else:
                            x1 = x0
                    elif y1 is None:
                        if a0 is not None:
                            # get intercept with desired x value
                            _, y1 = vector_intersection(
                                (x0, y0), a0, (x1, -1e6), 90, max_distance=2e6
                            )
                            if i_step == last_step_index:
                                a_final = a0
                        elif i_step == last_step_index:
                            if "exit_angle" in step:
                                y1 = y0
                            else:
                                _, y1 = vector_intersection(
                                    port2.center,
                                    port2.orientation,
                                    (x1, -1e6),
                                    90,
                                    max_distance=2e6,
                                )
                        else:
                            y1 = y0
                    elif a0 is None and i_step == last_step_index:
                        a_final = np.rad2deg(np.arctan2(y1 - y0, x1 - x0))

                    waypoints.append((x1, y1))
                    a0 = step.get("exit_angle", a_final)
                    angles.append(a0)
                    x0, y0 = x1, y1
                else:
                    raise ValueError(
                        f"Unable to process improperly or incompletely formed step routing command: {step}"
                    )
            has_explicit_end_angle = angles[-1] is not None

        if steps:
            waypoints.append(port2.center)
            bends = []
            prev_port = port1
            # go back over the waypoints and calculate unspecified angles and bends
            for i_step, step in enumerate(steps):
                # override the current cross section and connector, if applicable
                override_cs = steps[i_step].get("cross_section")
                override_connector = steps[i_step].get("connector")
                # override the current separation, if applicable
                override_separation = step.get("separation")
                if override_cs:
                    this_cs = get_cross_section(override_cs)
                    this_connector = auto_taper_connector
                else:
                    this_cs = cross_section
                    this_connector = connector_func
                if override_connector:
                    this_connector = get_connector(override_connector)

                # build up the sequence of references in the current segment of the route
                if i_step + 1 < len(angles):
                    # the angle going into the current step
                    # angle 0 should always be explicitly defined, from the first port
                    angle0 = angles[i_step]
                    # the angle going out of the current step
                    angle_next = angles[i_step + 1]
                    # if no angle was explicitly defined, let's calculate it here
                    if angle_next is None:
                        p1 = waypoints[i_step + 2]
                        p0 = waypoints[i_step + 1]
                        angle_next = np.rad2deg(
                            np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
                        )
                    dangle = difference_between_angles(angle_next, angle0)
                    if dangle == 180:
                        raise ValueError(
                            "Intermediate 180 degree bends are not currently supported!"
                        )
                    elif dangle == 0:
                        next_port = prev_port.flip()
                        next_port.center = waypoints[i_step + 1]
                        connection = this_connector(
                            prev_port, next_port, cross_section=this_cs
                        )
                        route_refs += connection
                        prev_port = next_port.flip()
                    else:
                        bend_component = _get_bend(
                            bend, angle=dangle, cross_section=cross_section
                        )
                        bend_ref = _place_bend(
                            bend_component,
                            position=waypoints[i_step + 1],
                            rotation=angle0,
                        )
                        bend_ref_ports = _get_bend_ports(bend_ref)
                        connection = this_connector(
                            prev_port, bend_ref_ports[0], cross_section=this_cs
                        )
                        route_refs += connection
                        route_refs.append(bend_ref)
                        bends.append(bend_ref)
                        prev_port = bend_ref_ports[1]
                    angles[i_step + 1] = angle_next
                    if is_primary_route:
                        if override_separation is not None:
                            this_separation = override_separation
                        elif separation is not None:
                            this_separation = separation
                        else:
                            this_separation = _get_minimum_separation(
                                connection, prev_port
                            )
                        segment_separations.append(this_separation)

            if has_explicit_end_angle:
                port1 = prev_port
            else:
                if _point_intersects_ray(
                    port2.center, port2.orientation, prev_port.center
                ):
                    final_connection = final_connector_func(
                        prev_port, port2, cross_section=final_cross_section
                    )
                    route_refs += final_connection
                    this_separation = _get_minimum_separation(
                        final_connection, prev_port
                    )
                    segment_separations.append(this_separation)
                else:
                    route_refs += _make_error_trace(
                        prev_port,
                        port2,
                        "Cannot complete final step of route! "
                        "Try setting an exit_angle in your final "
                        "step which intersects the vector of the destination port.",
                    )

        if not steps or has_explicit_end_angle:
            intersect = vector_intersection(
                port1.center,
                port1.orientation,
                port2.center,
                port2.orientation,
                raise_error=False,
            )
            report_segment_separation = None

            def _report_separations_w_steps(refs) -> None:
                segment_separations.append(_get_minimum_separation(refs))

            if steps:
                angles.insert(-1, port1.orientation)
                waypoints.insert(-1, intersect)
                report_segment_separation = _report_separations_w_steps
            bend_angle = difference_between_angles(
                port2.orientation + 180, port1.orientation
            )
            final_connection = _all_angle_connector(
                port1,
                port2,
                bend_angle=bend_angle,
                intersect=intersect,
                bend=bend,
                cross_section=cross_section,
                connector1=semi_final_connector_func,
                connector2=final_connector_func,
                cross_section1=semi_final_cross_section,
                cross_section2=final_cross_section,
                report_segment_separation=report_segment_separation,
            )
            route_refs += final_connection
            this_separation = _get_minimum_separation(final_connection, port1)
            segment_separations.append(this_separation)
        route_length = sum(r.info["length"] for r in route_refs)
        route = Route(
            references=route_refs,
            ports=(port1, port2),
            length=np.round(route_length, 3),
        )
        routes.append(route)
        is_primary_route = False
    return routes


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component("demo")

    mmi = gf.components.mmi2x2(width_mmi=10, gap_mmi=3)
    mmi1 = c << mmi
    mmi2 = c << mmi

    mmi2.move((100, 30))
    mmi2.rotate(30)

    routes = gf.routing.get_bundle_all_angle(
        mmi1.get_ports_list(orientation=0),
        [mmi2.ports["o2"], mmi2.ports["o1"]],
        connector=None,
    )
    for route in routes:
        c.add(route.references)
    c.show()

"""Routes bundles of ports (river routing).

get bundle is the generic river routing function
route_bundle calls different function depending on the port orientation.

 - route_bundle_same_axis: ports facing each other with arbitrary pitch on each side
 - route_bundle_corner: 90Deg / 270Deg between ports with arbitrary pitch
 - route_bundle_udirect: ports with direct U-turns
 - route_bundle_uindirect: ports with indirect U-turns

"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Literal, cast
from warnings import warn

import kfactory as kf
from kfactory.routing.generic import ManhattanRoute

import gdsfactory as gf
from gdsfactory.routing.auto_taper import add_auto_tapers
from gdsfactory.routing.sort_ports import get_port_x, get_port_y
from gdsfactory.typings import (
    STEP_DIRECTIVES,
    ComponentSpec,
    Coordinates,
    CrossSectionSpec,
    LayerSpec,
    LayerSpecs,
    LayerTransitions,
    Ports,
    Step,
)

OpticalManhattanRoute = ManhattanRoute

TOLERANCE = 1


def get_min_spacing(
    ports1: Ports,
    ports2: Ports,
    separation: float = 5.0,
    radius: float = 5.0,
    sort_ports: bool = True,
) -> float:
    """Returns the minimum amount of spacing in um required to create a fanout.

    Args:
        ports1: first list of ports.
        ports2: second list of ports.
        separation: minimum separation between two straights in um.
        radius: bend radius in um.
        sort_ports: sort the ports according to the axis.

    """
    axis = "X" if ports1[0].orientation in [0, 180] else "Y"
    j = 0
    min_j = 0
    max_j = 0
    if sort_ports:
        if axis in {"X", "x"}:
            sorted(ports1, key=get_port_y)
            sorted(ports2, key=get_port_y)
        else:
            sorted(ports1, key=get_port_x)
            sorted(ports2, key=get_port_x)

    for port1, port2 in zip(ports1, ports2, strict=False):
        if axis in {"X", "x"}:
            x1 = get_port_y(port1)
            x2 = get_port_y(port2)
        else:
            x1 = get_port_x(port1)
            x2 = get_port_x(port2)
        if x2 >= x1:
            j += 1
        else:
            j -= 1
        if j < min_j:
            min_j = j
        if j > max_j:
            max_j = j
    j = 0

    return (max_j - min_j) * separation + 2 * radius + 1.0


def route_bundle(
    component: gf.Component,
    ports1: Ports,
    ports2: Ports,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    separation: float = 3.0,
    bend: ComponentSpec = "bend_euler",
    sort_ports: bool = False,
    start_straight_length: float = 0,
    end_straight_length: float = 0,
    min_straight_taper: float = 100,
    taper: ComponentSpec | None = None,
    port_type: str | None = None,
    collision_check_layers: LayerSpecs | None = None,
    on_collision: Literal["error", "show_error"] | None = None,
    bboxes: Sequence[kf.kdb.DBox] | None = None,
    allow_width_mismatch: bool = False,
    radius: float | None = None,
    route_width: float | None = None,
    straight: ComponentSpec = "straight",
    auto_taper: bool = True,
    auto_taper_taper: ComponentSpec | None = None,
    waypoints: Coordinates | Sequence[gf.kdb.DPoint] | None = None,
    steps: Sequence[Step] | None = None,
    start_angles: float | list[float] | None = None,
    end_angles: float | list[float] | None = None,
    router: Literal["optical", "electrical"] | None = None,
    layer_transitions: LayerTransitions | None = None,
    layer_marker: LayerSpec | None = None,
    raise_on_error: bool = False,
) -> list[ManhattanRoute]:
    """Places a bundle of routes to connect two groups of ports.

    Routes connect a bundle of ports with a river router.
    Chooses the correct routing function depending on port angles.

    Args:
        component: component to add the routes to.
        ports1: list of starting ports.
        ports2: list of end ports.
        cross_section: CrossSection or function that returns a cross_section.
        layer: layer to use for the route.
        separation: bundle separation (center to center). Defaults to cross_section.width + cross_section.gap
        bend: function for the bend. Defaults to euler.
        sort_ports: sort port coordinates.
        start_straight_length: straight length at the beginning of the route. If None, uses default value for the routing CrossSection.
        end_straight_length: end length at the beginning of the route. If None, uses default value for the routing CrossSection.
        min_straight_taper: minimum length for tapering the straight sections.
        taper: function for tapering long straight waveguides beyond min_straight_taper. Defaults to None.
        port_type: type of port to place. Defaults to optical.
        collision_check_layers: list of layers to check for collisions.
        on_collision: action to take on collision. Defaults to None (ignore).
        bboxes: list of bounding boxes to avoid collisions.
        allow_width_mismatch: allow different port widths.
        radius: bend radius. If None, defaults to cross_section.radius.
        route_width: width of the route. If None, defaults to cross_section.width.
        straight: function for the straight. Defaults to straight.
        auto_taper: if True, auto-tapers ports to the cross-section of the route.
        auto_taper_taper: taper to use for auto-tapering. If None, uses the default taper for the cross-section.
        waypoints: list of waypoints to add to the route.
        steps: list of steps to add to the route.
        start_angles: list of start angles for the routes. Only used for electrical ports.
        end_angles: list of end angles for the routes. Only used for electrical ports.
        router: Set the type of router to use, either the optical one or the electrical one.
            If None, the router is optical unless the port_type is "electrical".
        layer_transitions: dictionary of layer transitions to use for the routing when auto_taper=True.
        layer_marker: layers to place markers on the route.
        raise_on_error: if True, raises an exception on routing error instead of adding error markers.

    .. plot::
        :include-source:

        import gdsfactory as gf

        dy = 200.0
        xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

        pitch = 10.0
        N = len(xs1)
        xs2 = [-20 + i * pitch for i in range(N // 2)]
        xs2 += [400 + i * pitch for i in range(N // 2)]

        a1 = 90
        a2 = a1 + 180

        ports1 = [gf.Port(name=f"top_{i}", center=(xs1[i], +0), width=0.5, orientation=a1, layer=(1,0)) for i in range(N)]
        ports2 = [gf.Port(name=f"bot_{i}", center=(xs2[i], dy), width=0.5, orientation=a2, layer=(1,0)) for i in range(N)]

        c = gf.Component()
        gf.routing.route_bundle(component=c, ports1=ports1, ports2=ports2, cross_section='strip', separation=5)
        c.plot()

    """
    if cross_section is None:
        if layer is None or route_width is None:
            raise ValueError(
                f"Either {cross_section=} or {layer=} and {route_width=} must be provided"
            )
    elif layer is not None:
        raise ValueError(
            f"Cannot have both {layer=} and {cross_section=} provided. Choose one."
        )

    c = component
    ports1_ = list(ports1)
    ports2_ = list(ports2)
    port_type = port_type or ports1_[0].port_type

    if cross_section is None:
        cross_section = partial(
            gf.cross_section.cross_section,
            layer=cast(LayerSpec, layer),
            width=cast(float, route_width),
            port_names=("e1", "e2") if port_type == "electrical" else ("o1", "o2"),
            port_types=(port_type, port_type),
        )

    if len(ports1_) != len(ports2_):
        raise ValueError(
            f"ports1={len(ports1_)} and ports2={len(ports2_)} must be equal"
        )
    if route_width:
        xs = gf.get_cross_section(cross_section, width=route_width)
    else:
        xs = gf.get_cross_section(cross_section)
    width = route_width or xs.width

    radius = radius or xs.radius
    taper_cell = gf.get_component(taper) if taper else None

    if collision_check_layers:
        collision_check_layer_enums = [
            gf.get_layer(layer) for layer in collision_check_layers
        ]
    else:
        collision_check_layer_enums = None

    bboxes = list(bboxes or [])

    if auto_taper and auto_taper_taper:
        warn(
            "Use of `auto_taper_taper` is deprecated. Please use `layer_transitions` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        taper_ = gf.get_component(auto_taper_taper)
        taper_o1 = taper_.ports[0].name
        taper_o2 = taper_.ports[1].name
        ports1_new = []
        ports2_new = []

        for p1, p2 in zip(ports1_, ports2_, strict=False):
            t1 = c << taper_
            t2 = c << taper_
            t1.connect(taper_o1, p1)
            t2.connect(taper_o1, p2)

            ports1_new.append(t1.ports[taper_o2])
            ports2_new.append(t2.ports[taper_o2])

        ports1_ = ports1_new
        ports2_ = ports2_new

        bbox1 = gf.kdb.DBox()
        bbox2 = gf.kdb.DBox()

        for port in ports1_:
            bbox1 += port.dcplx_trans.disp.to_p()

        for port in ports2_:
            bbox2 += port.dcplx_trans.disp.to_p()

        bboxes.append(bbox1)
        bboxes.append(bbox2)

    elif auto_taper:
        bbox1 = gf.kdb.DBox()
        bbox2 = gf.kdb.DBox()
        for port in ports1_:
            bbox1 += port.dcplx_trans.disp.to_p()

        for port in ports2_:
            bbox2 += port.dcplx_trans.disp.to_p()

        ports1_ = add_auto_tapers(
            component, ports1_, cross_section=xs, layer_transitions=layer_transitions
        )
        ports2_ = add_auto_tapers(
            component, ports2_, cross_section=xs, layer_transitions=layer_transitions
        )

        for port in ports1_:
            bbox1 += port.dcplx_trans.disp.to_p()

        for port in ports2_:
            bbox2 += port.dcplx_trans.disp.to_p()

        bboxes.append(bbox1)
        bboxes.append(bbox2)
        # component.shapes(component.kcl.layer(1,0)).insert(bbox)

    if steps and waypoints:
        raise ValueError("Cannot have both steps and waypoints")

    if steps:
        waypoints = []
        x, y = ports1_[0].center
        for d in steps:
            if not STEP_DIRECTIVES.issuperset(d):
                raise ValueError(
                    f"Invalid step directives: {list(d.keys() - STEP_DIRECTIVES)}."
                    f"Valid directives are {list(STEP_DIRECTIVES)}"
                )
            x = d.get("x", x) + d.get("dx", 0)
            y = d.get("y", y) + d.get("dy", 0)
            waypoints += [(x, y)]  # type: ignore[arg-type]
            if layer_marker:
                marker = component << gf.components.rectangle(
                    size=(10, 10), layer=layer_marker, centered=True
                )
                marker.center = (x, y)
    if waypoints is not None and not isinstance(waypoints[0], kf.kdb.DPoint):
        waypoints_: list[kf.kdb.DPoint] | None = [
            kf.kdb.DPoint(p[0], p[1])  # type: ignore[index]
            for p in waypoints
        ]
        if layer_marker and waypoints_ is not None:
            for p in waypoints_:
                marker = component << gf.components.rectangle(
                    size=(10, 10), layer=layer_marker, centered=True
                )
                marker.center = (p.x, p.y)
    else:
        waypoints_ = waypoints  # type: ignore[assignment]

    router = router or "electrical" if port_type == "electrical" else "optical"
    if router == "electrical":
        if cross_section is not None:
            xs = gf.get_cross_section(cross_section)
            layer_: gf.kdb.LayerInfo | None = gf.kcl.get_info(
                gf.get_layer(xs.sections[0].layer)
            )
        else:
            layer_ = None
        try:
            route = kf.routing.electrical.route_bundle(
                component,
                ports1_,
                ports2_,
                separation=separation,
                starts=start_straight_length,
                ends=end_straight_length,
                collision_check_layers=[
                    c.kcl.layout.get_info(layer)
                    for layer in collision_check_layer_enums
                ]
                if collision_check_layer_enums is not None
                else None,
                on_collision=on_collision,
                bboxes=bboxes,
                route_width=width,
                sort_ports=sort_ports,
                waypoints=waypoints_,
                end_angles=end_angles,
                start_angles=start_angles,
                place_layer=layer_,
            )
        except Exception as e:
            if raise_on_error:
                raise e
            gf.logger.error(f"Error in route_bundle: {e}")
            layer_error_path = gf.get_layer_info(gf.CONF.layer_error_path)
            route = kf.routing.electrical.route_bundle(
                component,
                ports1_,
                ports2_,
                separation=separation,
                starts=start_straight_length,
                ends=end_straight_length,
                on_collision=on_collision,
                bboxes=bboxes,
                route_width=width,
                sort_ports=sort_ports,
                end_angles=end_angles,
                start_angles=start_angles,
                place_layer=layer_error_path,
            )

            if waypoints and waypoints_ is not None:
                layer_marker = gf.CONF.layer_error_path
                for p in waypoints_:
                    marker = component << gf.components.rectangle(
                        size=(10, 10), layer=layer_marker, centered=True
                    )
                    marker.center = (p.x, p.y)

        return route

    bend90 = (
        bend
        if isinstance(bend, gf.Component)
        else gf.get_component(
            bend, cross_section=cross_section, radius=radius, width=width
        )
    )

    def straight_um(width: float, length: float) -> gf.Component:
        return gf.get_component(
            straight, length=length, cross_section=cross_section, width=width
        )

    try:
        route = kf.routing.optical.route_bundle(
            component,
            ports1_,
            ports2_,
            separation=separation,
            straight_factory=straight_um,
            bend90_cell=bend90,
            taper_cell=taper_cell,
            starts=start_straight_length,
            ends=end_straight_length,
            min_straight_taper=min_straight_taper,
            place_port_type=port_type,
            collision_check_layers=[
                c.kcl.layout.get_info(layer) for layer in collision_check_layer_enums
            ]
            if collision_check_layer_enums
            else None,
            on_collision=on_collision,
            allow_width_mismatch=allow_width_mismatch,
            bboxes=list(bboxes or []),
            route_width=width,
            sort_ports=sort_ports,
            waypoints=waypoints_,
            end_angles=end_angles,
            start_angles=start_angles,
        )
    except Exception as e:
        if raise_on_error:
            raise e
        gf.logger.error(f"Error in route_bundle: {e}")
        layer_error_path = gf.get_layer_info(gf.CONF.layer_error_path)
        route = kf.routing.electrical.route_bundle(
            component,
            ports1_,
            ports2_,
            separation=separation,
            starts=start_straight_length,
            ends=end_straight_length,
            on_collision=on_collision,
            bboxes=bboxes,
            route_width=width,
            sort_ports=sort_ports,
            end_angles=end_angles,
            start_angles=start_angles,
            place_layer=layer_error_path,
        )

        if waypoints and waypoints_ is not None:
            layer_marker = gf.CONF.layer_error_path
            for p in waypoints_:
                marker = component << gf.components.rectangle(
                    size=(10, 10), layer=layer_marker, centered=True
                )
                marker.center = (p.x, p.y)

    return route


route_bundle_electrical = partial(
    route_bundle,
    bend="wire_corner",
    allow_width_mismatch=True,
)

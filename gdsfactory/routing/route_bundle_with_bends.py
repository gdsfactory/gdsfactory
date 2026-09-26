"""Bundle routing with ordered bend sequences."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import partial
from typing import Any, Literal, cast
from warnings import warn

import kfactory as kf
from kfactory.routing.generic import ManhattanRoute
from kfactory.routing.optical import PathLengthConfig, place_manhattan
from kfactory.schematic import Constraint

import gdsfactory as gf
from gdsfactory.config import CONF
from gdsfactory.routing.auto_taper import add_auto_tapers
from gdsfactory.routing.resolve_pins import resolve_pins
from gdsfactory.routing.route_bundle import _ensure_manhattan_waypoints
from gdsfactory.typings import (
    STEP_DIRECTIVES,
    ComponentSpec,
    Coordinates,
    CrossSectionSpec,
    LayerSpec,
    LayerSpecs,
    LayerTransitions,
    Pin,
    Port,
    Ports,
    Step,
)


def _route_exact_waypoint_bend_sequence(
    component: gf.Component,
    port1: gf.Port,
    port2: gf.Port,
    waypoints: Sequence[kf.kdb.DPoint],
    width: float,
    cross_section: CrossSectionSpec,
    straight: ComponentSpec,
    bend90: gf.Component,
    bend90_cells: Sequence[gf.Component],
    taper_cell: gf.Component | None,
    port_type: str,
    min_straight_taper: float,
    allow_width_mismatch: bool | None,
    allow_layer_mismatch: bool | None,
    allow_type_mismatch: bool | None,
) -> ManhattanRoute:
    """Route a single explicit waypoint path with an ordered bend sequence."""

    def straight_dbu(width: int, length: int, **kwargs: Any) -> gf.Component:
        xs = kwargs.pop("cross_section", cross_section)
        return gf.get_component(
            straight,
            length=component.kcl.to_um(length),
            cross_section=xs,
            width=component.kcl.to_um(width),
            **kwargs,
        )

    pts = [component.kcl.to_dbu(kf.kdb.DPoint(*port1.center))]
    pts.extend(point.to_itype(component.kcl.dbu) for point in waypoints)
    pts.append(component.kcl.to_dbu(kf.kdb.DPoint(*port2.center)))

    return place_manhattan(
        component.to_itype(),
        p1=port1.to_itype(),
        p2=port2.to_itype(),
        pts=pts,
        route_width=component.kcl.to_dbu(width),
        straight_factory=straight_dbu,
        bend90_cell=bend90.to_itype(),
        bend90_cells=[bend_cell.to_itype() for bend_cell in bend90_cells],
        taper_cell=taper_cell.to_itype() if taper_cell else None,
        port_type=port_type,
        min_straight_taper=component.kcl.to_dbu(min_straight_taper),
        allow_width_mismatch=allow_width_mismatch,
        allow_layer_mismatch=allow_layer_mismatch,
        allow_type_mismatch=allow_type_mismatch,
    )


def route_bundle_with_bends(
    component: gf.Component,
    ports1: Port | Ports | list[Pin] | None = None,
    ports2: Port | Ports | list[Pin] | None = None,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    separation: float = 3.0,
    bend: ComponentSpec | Sequence[ComponentSpec] = "bend_euler",
    sort_ports: bool = False,
    start_straight_length: float = 0,
    end_straight_length: float = 0,
    min_straight_taper: float = 100,
    taper: ComponentSpec | None = None,
    port_type: str | None = None,
    collision_check_layers: LayerSpecs | None = None,
    on_collision: Literal["error", "show_error", "warning", "ignore"] | None = None,
    on_placer_error: Literal["error", "show_error", "warning", "ignore"] | None = None,
    bboxes: Sequence[kf.kdb.DBox] | None = None,
    allow_width_mismatch: bool | None = None,
    allow_layer_mismatch: bool | None = None,
    allow_type_mismatch: bool | None = None,
    radius: float | None = None,
    route_width: float | None = None,
    straight: ComponentSpec = "straight",
    sbend: ComponentSpec | None = None,
    auto_taper: bool = True,
    auto_taper_taper: ComponentSpec | None = None,
    waypoints: Coordinates | Sequence[gf.kdb.DPoint] | None = None,
    steps: Sequence[Step] | None = None,
    start_angles: float | list[float] | None = None,
    end_angles: float | list[float] | None = None,
    router: Literal["optical", "electrical"] | None = None,
    layer_transitions: LayerTransitions | None = None,
    show_waypoints: bool = False,
    layer_marker: LayerSpec | None = None,
    raise_on_error: bool | None = None,
    path_length_matching_config: PathLengthConfig | None = None,
    constraints: Sequence[Constraint] | None = None,
    layer_label: LayerSpec | None = None,
    port1: Port | None = None,
    port2: Port | None = None,
    name: str | None = None,
) -> list[ManhattanRoute]:
    """Places a bundle of routes and supports ordered bend sequences."""
    name = name or "unnamed_route_bundle"
    if on_collision is None:
        on_collision = CONF.on_collision
    if on_placer_error is None:
        on_placer_error = CONF.on_placer_error

    if raise_on_error is None:
        raise_on_error = CONF.raise_on_error

    if port1 is not None:
        if ports1 is not None:
            raise ValueError("Cannot specify both ports1 and port1")
        ports1 = port1
    if port2 is not None:
        if ports2 is not None:
            raise ValueError("Cannot specify both ports2 and port2")
        ports2 = port2

    if ports1 is None or ports2 is None:
        raise ValueError("ports1 and ports2 are required")

    if isinstance(ports1, kf.DPort):
        ports1 = [ports1]
    if isinstance(ports2, kf.DPort):
        ports2 = [ports2]

    port_list1 = list(ports1)
    port_list2 = list(ports2)

    if port_list1 and isinstance(port_list1[0], kf.DPin):
        if not (port_list2 and isinstance(port_list2[0], kf.DPin)):
            raise TypeError(
                "Cannot mix Pins and Ports. "
                "If ports1 contains Pins, ports2 must also contain Pins."
            )
        port_list1, port_list2 = resolve_pins(
            cast(list[Pin], port_list1), cast(list[Pin], port_list2)
        )
    elif port_list2 and isinstance(port_list2[0], kf.DPin):
        raise TypeError(
            "Cannot mix Pins and Ports. "
            "If ports2 contains Pins, ports1 must also contain Pins."
        )

    if show_waypoints and layer_marker is None:
        layer_marker = gf.CONF.layer_marker

    component = gf.Component(base=component.base)  # type: ignore[call-overload]
    ports1_resolved = [gf.Port(base=p1.base) for p1 in cast(list[kf.DPort], port_list1)]
    ports2_resolved = [gf.Port(base=p2.base) for p2 in cast(list[kf.DPort], port_list2)]

    if router:
        warnings.warn(
            f"The argument {router=} is ignored and will be removed in a future release.",
            stacklevel=2,
        )

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
    ports1_ = ports1_resolved
    ports2_ = ports2_resolved
    port_type = port_type or ports1_[0].port_type

    if cross_section is None:
        cross_section = partial(
            gf.cross_section.cross_section,
            layer=cast("LayerSpec", layer),
            width=cast("float", route_width),
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
        ports1_new: list[gf.Port] = []
        ports2_new: list[gf.Port] = []

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

    if steps and waypoints:
        raise ValueError("Provide only one of steps or waypoints")

    if steps:
        waypoints = []
        x, y = ports1_[0].center
        for d in steps:
            if isinstance(d, dict):
                if not STEP_DIRECTIVES.issuperset(d):
                    raise ValueError(
                        f"Invalid step directives: {list(d.keys() - STEP_DIRECTIVES)}."
                        f"Valid directives are {list(STEP_DIRECTIVES)}"
                    )
                x = d.get("x", x) + d.get("dx", 0)
                y = d.get("y", y) + d.get("dy", 0)
            else:
                raise ValueError(
                    f"Invalid step {d!r}. Each step must be a dict with keys (x, y, dx, dy)."
                )
            waypoints += [(x, y)]  # type: ignore[arg-type]
            if layer_marker:
                marker = component << gf.components.rectangle(
                    size=(10, 10), layer=layer_marker, centered=True
                )
                marker.center = (x, y)

    if waypoints is not None and steps and len(waypoints) < 2:
        x, y = waypoints[-1][0], waypoints[-1][1]  # type: ignore[index]
        x1, y1 = ports1_[0].center
        port2 = ports2_[0]
        x2, y2 = port2.center
        orientation = port2.orientation
        if orientation is not None and int(orientation) in {0, 180}:
            yt = y1 + (y2 - y1) / 3
            ytt = y1 + 2 * (y2 - y1) / 3
            waypoints = [(x, yt), (x, ytt)]
        elif orientation is not None and int(orientation) in {90, 270}:
            xt = x1 + (x2 - x1) / 3
            xtt = x1 + 2 * (x2 - x1) / 3
            waypoints = [(xt, y), (xtt, y)]

    waypoints_: list[kf.kdb.DPoint] | None
    if waypoints is None:
        waypoints_ = None
    elif len(waypoints) == 0:
        waypoints_ = []
    elif not isinstance(waypoints[0], kf.kdb.DPoint):
        waypoints_ = [
            kf.kdb.DPoint(p[0], p[1])  # type: ignore[index]
            for p in waypoints
        ]
    else:
        waypoints_ = [cast("kf.kdb.DPoint", p) for p in waypoints]

    if layer_marker and waypoints_ is not None:
        for p in waypoints_:
            marker = component << gf.components.rectangle(
                size=(10, 10), layer=layer_marker, centered=True
            )
            marker.center = (p.x, p.y)

    if waypoints_ is not None and len(waypoints_) >= 2:
        waypoints_ = _ensure_manhattan_waypoints(waypoints_, start_port=ports1_[0])

    if isinstance(bend, Sequence) and not isinstance(bend, str):
        bend_sequence = list(bend)
        if not bend_sequence:
            raise ValueError("bend sequence must contain at least one bend spec")

        def _resolve_bend_spec(bend_spec: ComponentSpec) -> gf.Component:
            if isinstance(bend_spec, gf.Component):
                return bend_spec

            bend_kwargs: dict[str, Any] = {
                "cross_section": cross_section,
                "width": width,
            }
            if not (
                isinstance(bend_spec, partial)
                and bend_spec.keywords
                and "radius" in bend_spec.keywords
            ):
                bend_kwargs["radius"] = radius

            return gf.get_component(bend_spec, **bend_kwargs)

        bend90_cells = [_resolve_bend_spec(bend_spec) for bend_spec in bend_sequence]
        bend90 = bend90_cells[0]
    else:
        bend90_cells = None
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

    if sbend:

        def _sbend(
            c: gf.kf.ProtoTKCell[Any], offset: float, length: float, width: float
        ) -> gf.kf.DInstanceGroup:
            sb = gf.get_component(
                sbend,
                cross_section=cross_section,
                width=width,
                size=(length, offset),
            )
            sb_ref = component << sb
            return gf.kf.DInstanceGroup(insts=[sb_ref], ports=list(sb_ref.ports))

    if path_length_matching_config is not None and constraints is not None:
        raise ValueError(
            "path_length_matching_config and constraints are mutually exclusive. "
            "Pass a kf.schematic.PathLengthMatch constraint via constraints instead."
        )

    if path_length_matching_config is not None:
        route_constraints: list[Constraint] = [
            kf.schematic.PathLengthMatch(
                route_names=[name],
                instance_names=[],
                on_failure=None,
                loops=path_length_matching_config.get("loops", 1),
                loop_side=path_length_matching_config.get("loop_side", -1),
                element=path_length_matching_config.get("element", -1),
                loop_position=path_length_matching_config.get("loop_position", -1),
                length=path_length_matching_config.get("total_length"),
                all=True,
            )
        ]
    else:
        route_constraints = list(constraints or [])

    try:
        kf_on_collision = on_collision
        if kf_on_collision == "warning":
            kf_on_collision = "error"
        elif kf_on_collision == "ignore":
            kf_on_collision = None

        kf_on_placer_error = on_placer_error
        if kf_on_placer_error == "warning":
            kf_on_placer_error = "error"
        elif kf_on_placer_error == "ignore":
            kf_on_placer_error = None

        if (
            bend90_cells is not None
            and waypoints_ is not None
            and len(ports1_) == 1
            and len(ports2_) == 1
            and sbend is None
        ):
            route = [
                _route_exact_waypoint_bend_sequence(
                    component=component,
                    port1=ports1_[0],
                    port2=ports2_[0],
                    waypoints=waypoints_,
                    width=width,
                    cross_section=cross_section,
                    straight=straight,
                    bend90=bend90,
                    bend90_cells=bend90_cells,
                    taper_cell=taper_cell,
                    port_type=port_type,
                    min_straight_taper=min_straight_taper,
                    allow_width_mismatch=allow_width_mismatch,
                    allow_layer_mismatch=allow_layer_mismatch,
                    allow_type_mismatch=allow_type_mismatch,
                )
            ]
        else:
            route = kf.routing.optical.route_bundle(
                component,
                ports1_,
                ports2_,
                separation=separation,
                straight_factory=straight_um,
                bend90_cell=bend90,
                bend90_cells=bend90_cells,
                taper_cell=taper_cell,
                starts=start_straight_length,
                ends=end_straight_length,
                min_straight_taper=min_straight_taper,
                place_port_type=port_type,
                collision_check_layers=[
                    c.kcl.layout.get_info(layer)
                    for layer in collision_check_layer_enums
                ]
                if collision_check_layer_enums
                else None,
                on_collision=kf_on_collision,
                on_placer_error=kf_on_placer_error,
                allow_width_mismatch=allow_width_mismatch,
                allow_layer_mismatch=allow_layer_mismatch,
                allow_type_mismatch=allow_type_mismatch,
                bboxes=list(bboxes or []),
                route_width=width,
                sort_ports=sort_ports,
                waypoints=waypoints_,
                end_angles=end_angles,
                start_angles=start_angles,
                constraints=route_constraints,
                sbend_factory=_sbend if sbend else None,
            )
    except Exception as e:
        if raise_on_error:
            if "kdb.Trans" in str(e):
                raise ValueError("You need at least 2 waypoints or steps.") from e
            if "non-manhattan" in str(e):
                raise ValueError(
                    "Waypoints need to be Manhattan (axis-aligned) coordinates."
                ) from e
            raise

        if "kdb.Trans" in str(e):
            e = ValueError("You need at least 2 waypoints or steps.")
        elif "non-manhattan" in str(e):
            e = ValueError("Waypoints need to be Manhattan (axis-aligned) coordinates.")
        gf.logger.error(f"Error in route_bundle: {e}")
        warn(f"Routing failed: {e}", stacklevel=2)
        layer_error_path = gf.get_layer_info(gf.CONF.layer_error_path)
        route = kf.routing.electrical.route_bundle(
            component,
            ports1_,
            ports2_,
            separation=separation,
            starts=start_straight_length,
            ends=end_straight_length,
            on_collision=None,
            on_placer_error=None,
            bboxes=bboxes,
            route_width=width,
            sort_ports=sort_ports,
            waypoints=waypoints_,
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

    if layer_label:
        for route_i in route:
            c.add_label(
                text=f"{route_i.length:.3f}",
                layer=layer_label,
                position=route_i.instances[0].dcenter,
            )

    return route
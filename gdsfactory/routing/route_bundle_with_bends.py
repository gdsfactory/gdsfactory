"""Bundle routing with an ordered sequence of bends on each route."""

from __future__ import annotations

from collections.abc import Sequence

import kfactory as kf
from kfactory.routing.generic import ManhattanRoute, get_radius
from kfactory.routing.manhattan import route_smart

import gdsfactory as gf
from gdsfactory.routing.place_manhattan_with_bends import place_manhattan_with_bends
from gdsfactory.routing.utils import validate_bend90
from gdsfactory.typings import ComponentSpec, Coordinates, CrossSectionSpec, Ports


def route_bundle_with_bends(
    component: gf.Component,
    ports1: Ports,
    ports2: Ports,
    *,
    bends: Sequence[ComponentSpec],
    cross_section: CrossSectionSpec,
    separation: float = 3.0,
    waypoints: Coordinates | None = None,
) -> list[ManhattanRoute]:
    """Route port pairs, using ``bends`` in order on each route.

    The bend sequence restarts for every port pair. If a route has more corners
    than bends, its last bend is reused. Waypoints are shared bundle guides,
    as in :func:`route_bundle`, rather than exact corners on every route.
    """
    start_ports = list(ports1)
    end_ports = list(ports2)
    if len(start_ports) != len(end_ports):
        raise ValueError("ports1 and ports2 must have the same length")
    if not start_ports:
        return []
    if not bends:
        raise ValueError("bends must contain at least one bend")
    if separation <= 0:
        raise ValueError("separation must be positive")

    xs = gf.get_cross_section(cross_section)
    port_type = start_ports[0].port_type
    if any(p.port_type != port_type for p in (*start_ports, *end_ports)):
        raise ValueError("All route ports must have the same port type")
    if any(not p.is_symmetric() for p in (*start_ports, *end_ports)):
        raise ValueError("Asymmetric ports are not supported")

    bend_cells = [
        bend
        if isinstance(bend, gf.Component)
        else gf.get_component(bend, cross_section=xs)
        for bend in bends
    ]
    for bend in bend_cells:
        validate_bend90(bend, port_type, bend_cells[0])
    bend_radius = max(
        get_radius(bend.ports.filter(port_type=port_type)) for bend in bend_cells
    )

    c = component.to_itype()

    def straight_factory(width: int, length: int) -> gf.Component:
        return gf.get_component(
            "straight",
            length=c.kcl.to_um(length),
            width=c.kcl.to_um(width),
            cross_section=xs,
        ).to_itype()

    waypoint_points = (
        [
            p.to_itype(c.kcl.dbu)
            if isinstance(p, kf.kdb.DPoint)
            else c.kcl.to_dbu(kf.kdb.DPoint(p[0], p[1]))
            for p in waypoints
        ]
        if waypoints is not None
        else None
    )

    return kf.routing.generic.route_bundle(
        c=c,
        start_ports=[p.to_itype().base for p in start_ports],
        end_ports=[p.to_itype().base for p in end_ports],
        route_width=c.kcl.to_dbu(xs.width),
        on_collision="error",
        on_placer_error="error",
        routing_function=route_smart,
        routing_kwargs={
            "bend90_radius": bend_radius,
            "separation": c.kcl.to_dbu(separation),
            "sort_ports": False,
            "waypoints": waypoint_points,
            "bbox_routing": "minimal",
            "bboxes": [],
            "allow_sbend": False,
        },
        placer_function=place_manhattan_with_bends,
        placer_kwargs={
            "bend90_cells": [bend.to_itype() for bend in bend_cells],
            "straight_factory": straight_factory,
            "route_width": c.kcl.to_dbu(xs.width),
            "port_type": port_type,
        },
    )

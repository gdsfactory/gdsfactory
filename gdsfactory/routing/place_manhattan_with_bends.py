"""Place a Manhattan backbone with a different bend at each corner."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import kfactory as kf
from kfactory import kdb
from kfactory.conf import config
from kfactory.factories import StraightFactoryDBU
from kfactory.kcell import ProtoTKCell
from kfactory.port import Port
from kfactory.routing.generic import ManhattanRoute
from kfactory.routing.optical import (
    _bend90_geometry,
    _place_straight,
    _place_tapered_straight,
    vec_angle,
)


def place_manhattan_with_bends(
    c: ProtoTKCell[Any],
    p1: Port,
    p2: Port,
    pts: Sequence[kdb.Point],
    *,
    bend90_cells: Sequence[ProtoTKCell[Any]],
    straight_factory: StraightFactoryDBU,
    route_width: int | None = None,
    taper_cell: ProtoTKCell[Any] | None = None,
    port_type: str = "optical",
    min_straight_taper: int = 0,
    allow_small_routes: bool = False,
    allow_width_mismatch: bool | None = None,
    allow_layer_mismatch: bool | None = None,
    allow_type_mismatch: bool | None = None,
    purpose: str | None = "routing",
) -> ManhattanRoute:
    """Place bends in order along an exact backbone, repeating the last bend.

    Coordinates and lengths are in database units. Each interior point is a
    90-degree corner. As in kfactory's ``place_manhattan``, the caller supplies
    the backbone; this function places its bends and connecting straight sections.
    """
    if not bend90_cells:
        raise ValueError("bend90_cells must contain at least one bend")
    if allow_width_mismatch is None:
        allow_width_mismatch = config.allow_width_mismatch
    if allow_layer_mismatch is None:
        allow_layer_mismatch = config.allow_layer_mismatch
    if allow_type_mismatch is None:
        allow_type_mismatch = config.allow_type_mismatch

    cell = kf.KCell(base=c.base)
    bends = [kf.KCell(base=bend.base) for bend in bend90_cells]
    geometries = [_bend90_geometry(bend, port_type) for bend in bends]
    radii: list[int] = []
    for bend_port1, bend_port2, corner in geometries:
        corner.mirror = bend_port1.mirror
        radii.append(
            round(
                max(
                    (bend_port1.trans.disp - corner.disp).length(),
                    (bend_port2.trans.disp - corner.disp).length(),
                )
            )
        )

    start_port = p1.copy()
    end_port = p2.copy()
    start_port.name = "route_start"
    start_port.angle = (start_port.angle + 2) % 4
    end_port.name = "route_end"
    end_port.angle = (end_port.angle + 2) % 4
    route = ManhattanRoute(
        backbone=list(pts),
        start_port=start_port,
        end_port=end_port,
        instances=[],
        bend90_radius=max(radii),
    )

    taper_ports = None
    if taper_cell is not None:
        taper_cell = kf.KCell(base=taper_cell.base)
        taper_ports_ = [p for p in taper_cell.ports if p.port_type == port_type]
        if len(taper_ports_) != 2 or (
            (taper_ports_[1].trans.angle + 2) % 4 != taper_ports_[0].trans.angle
        ):
            raise AttributeError("Taper must have two opposite routing ports")
        if taper_ports_[1].width == geometries[0][0].width:
            taper_ports = taper_ports_[1], taper_ports_[0]
        elif taper_ports_[0].width == geometries[0][0].width:
            taper_ports = taper_ports_[0], taper_ports_[1]
        else:
            raise AttributeError("Taper width must match the bend port width")
        route.taper_length = int(
            (taper_ports[0].trans.disp - taper_ports[1].trans.disp).length()
        )

    width = route_width or p1.width

    def segment(start: Port, end: Port) -> tuple[Port, Port]:
        length = int((end.trans.disp - start.trans.disp).length())
        if (
            taper_cell is not None
            and taper_ports is not None
            and length >= 2 * route.taper_length + min_straight_taper
        ):
            return _place_tapered_straight(
                c=cell,
                straight_factory=straight_factory,
                taper_cell=taper_cell,
                purpose=purpose,
                route=route,
                p1=start,
                p2=end,
                route_width=route_width,
                taper_ports=taper_ports,
                port_type=port_type,
                allow_width_mismatch=allow_width_mismatch,
                allow_layer_mismatch=allow_layer_mismatch,
                allow_type_mismatch=allow_type_mismatch,
            )
        return _place_straight(
            c=cell,
            straight_factory=straight_factory,
            purpose=purpose,
            w=width,
            route=route,
            p1=start,
            p2=end,
            route_width=route_width,
            port_type=port_type,
            allow_width_mismatch=allow_width_mismatch,
            allow_layer_mismatch=allow_layer_mismatch,
            allow_type_mismatch=allow_type_mismatch,
        )

    if len(pts) < 2:
        return route
    if len(pts) == 2:
        first, last = segment(start_port.copy_polar(), end_port.copy_polar())
        first.name = "route_start"
        last.name = "route_end"
        route.start_port = p1
        route.end_port = p2
        return route

    previous_bend_port = p1
    for i in range(1, len(pts) - 1):
        bend_index = min(i - 1, len(bends) - 1)
        bend = bends[bend_index]
        bend_port1, bend_port2, corner = geometries[bend_index]
        radius = radii[bend_index]
        previous_radius = radii[min(i - 2, len(bends) - 1)] if i > 1 else 0
        incoming = pts[i] - pts[i - 1]
        outgoing = pts[i + 1] - pts[i]
        if (incoming.x != 0 and incoming.y != 0) or (
            outgoing.x != 0 and outgoing.y != 0
        ):
            raise ValueError("Backbone segments must be Manhattan")
        if (vec_angle(outgoing) - vec_angle(incoming)) % 4 not in (1, 3):
            raise ValueError("Each interior backbone point must be a 90-degree turn")
        if not allow_small_routes and incoming.length() < previous_radius + radius:
            raise ValueError("Not enough room between consecutive bends")
        if i == len(pts) - 2 and not allow_small_routes:
            if outgoing.length() < radius:
                raise ValueError("Not enough room after the final bend")

        instance = cell << bend
        instance.purpose = purpose
        mirror = (vec_angle(outgoing) - vec_angle(incoming)) % 4 != 3
        angle = (vec_angle(incoming) + 2) % 4
        instance.transform(
            kdb.Trans(angle, mirror, pts[i].x, pts[i].y) * corner.inverted()
        )
        bend_in = instance.ports[bend_port1.name]
        bend_out = instance.ports[bend_port2.name]
        length = int((bend_in.trans.disp - previous_bend_port.trans.disp).length())
        if length > 0:
            first, _ = segment(previous_bend_port, bend_in)
            if i == 1:
                route.start_port = first
        route.instances.append(instance)
        route.n_bend90 += 1
        previous_bend_port = bend_out

    length = int((previous_bend_port.trans.disp - p2.trans.disp).length())
    if length > 0:
        _, last = segment(previous_bend_port, p2)
        route.end_port = last.copy()
    else:
        route.end_port = previous_bend_port.copy()
    route.start_port.name = "route_start"
    route.end_port.name = "route_end"
    return route

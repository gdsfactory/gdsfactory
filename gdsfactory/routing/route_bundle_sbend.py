from __future__ import annotations

from typing import Any

import kfactory as kf
from kfactory.routing.generic import ManhattanRoute

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.routing.auto_taper import add_auto_tapers
from gdsfactory.routing.sort_ports import sort_ports as sort_ports_function
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    LayerTransitions,
    Port,
    Ports,
)


def route_bundle_sbend(
    component: Component,
    ports1: Port | Ports,
    ports2: Port | Ports,
    bend_s: ComponentSpec = "bend_s",
    sort_ports: bool = True,
    enforce_port_ordering: bool = True,
    allow_width_mismatch: bool | None = None,
    allow_layer_mismatch: bool | None = None,
    allow_type_mismatch: bool | None = None,
    port_name: str = "o1",
    use_port_width: bool = True,
    auto_taper: bool = True,
    cross_section: CrossSectionSpec | None = None,
    layer_transitions: LayerTransitions | None = None,
    **kwargs: Any,
) -> list[ManhattanRoute]:
    """Places sbend routes from ports1 to ports2.

    Args:
        component: to place the sbend routes.
        ports1: start ports.
        ports2: end ports.
        bend_s: Sbend component.
        sort_ports: sort ports.
        enforce_port_ordering: enforces port ordering.
        allow_width_mismatch: allows width mismatch.
        allow_layer_mismatch: allows layer mismatch.
        allow_type_mismatch: allows type mismatch.
        port_name: name of the port to connect to the sbend.
        use_port_width: if True, use the width of the port to set the width of the sbend.
        auto_taper: if True, auto-tapers ports to the cross-section of the route.
        cross_section: cross-section to use for auto-tapering. Required when auto_taper=True.
        layer_transitions: dictionary of layer transitions for auto-tapering.
        kwargs: cross_section settings.

    """
    # Wrap single ports in lists
    if isinstance(ports1, kf.DPort):
        ports1 = [ports1]
    if isinstance(ports2, kf.DPort):
        ports2 = [ports2]

    if sort_ports:
        ports1, ports2 = sort_ports_function(
            ports1, ports2, enforce_port_ordering=enforce_port_ordering
        )

    if auto_taper and cross_section is not None:
        xs_ = gf.get_cross_section(cross_section)
        ports1 = add_auto_tapers(
            component,
            list(ports1),
            cross_section=xs_,
            layer_transitions=layer_transitions,
        )
        ports2 = add_auto_tapers(
            component,
            list(ports2),
            cross_section=xs_,
            layer_transitions=layer_transitions,
        )

    routes = []

    for p1, p2 in zip(list(ports1), list(ports2), strict=False):
        orthogonality_error = abs(abs(p1.orientation - p2.orientation) - 180)
        if orthogonality_error > 0.1:
            raise ValueError(
                f"Ports need to have orthogonal orientation {orthogonality_error}\n"
                f"port1 = {p1.orientation} deg and port2 = {p2.orientation}"
            )

        ys = p2.center[1] - p1.center[1]
        xs = p2.center[0] - p1.center[0]

        if p1.orientation in [0, 180]:
            xsize = xs
            ysize = ys
        elif p1.orientation == 90:
            xsize = ys
            ysize = -xs

        elif p1.orientation == 270:
            xsize = -ys
            ysize = xs

        bend_kwargs = dict(**kwargs)
        if cross_section is not None:
            bend_kwargs["cross_section"] = cross_section
        if use_port_width:
            bend = gf.get_component(
                bend_s, size=(xsize, ysize), width=p1.width, **bend_kwargs
            )
        else:
            bend = gf.get_component(bend_s, size=(xsize, ysize), **bend_kwargs)
        sbend = component << bend
        sbend.connect(
            port_name,
            p1,
            allow_width_mismatch=allow_width_mismatch,
            allow_layer_mismatch=allow_layer_mismatch,
            allow_type_mismatch=allow_type_mismatch,
        )

        route = ManhattanRoute(
            backbone=[],
            start_port=p1.to_itype(),
            end_port=p2.to_itype(),
            instances=[],
            bend90_radius=round(bend.info.get("min_bend_radius", 0)),
        )
        routes.append(route)
    return routes

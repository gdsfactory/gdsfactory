"""`route_single` places a Manhattan route between two ports.

`route_single` only works for an individual routes. For routing groups of ports you need to use `route_bundle` instead

To make a route, you need to supply:

 - input port
 - output port
 - bend
 - straight
 - taper to taper to wider straights and reduce straight loss (Optional)

To generate a route:

 1. Generate the backbone of the route.
 This is a list of manhattan coordinates that the route would pass through
 if it used only sharp bends (right angles)

 2. Replace the corners by bend references
 (with rotation and position computed from the manhattan backbone)

 3. Add tapers if needed and if space permits

 4. generate straight portions in between tapers or bends

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import kfactory as kf
from kfactory.routing.electrical import route_elec
from kfactory.routing.generic import ManhattanRoute
from kfactory.routing.optical import place_manhattan, route

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.routing.auto_taper import add_auto_tapers
from gdsfactory.typings import (
    STEP_DIRECTIVES,
    ComponentSpec,
    CrossSectionSpec,
    LayerSpec,
    LayerTransitions,
    Port,
    Step,
    WayPoints,
)


def route_single(
    component: Component,
    port1: Port,
    port2: Port,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    start_straight_length: float = 0.0,
    end_straight_length: float = 0.0,
    waypoints: WayPoints | None = None,
    steps: Sequence[Step] | None = None,
    port_type: str | None = None,
    allow_width_mismatch: bool = False,
    radius: float | None = None,
    route_width: float | None = None,
    auto_taper: bool = True,
    on_error: Literal["error"] | None = None,
    layer_transitions: LayerTransitions | None = None,
) -> ManhattanRoute:
    """Returns a Manhattan Route between 2 ports.

    The references are straights, bends and tapers.

    Args:
        component: to place the route into.
        port1: start port.
        port2: end port.
        cross_section: spec.
        layer: layer spec.
        bend: bend spec.
        straight: straight spec.
        start_straight_length: length of starting straight.
        end_straight_length: length of end straight.
        waypoints: optional list of points to pass through.
        steps: optional list of steps to pass through.
        port_type: port type to route.
        allow_width_mismatch: allow different port widths.
        radius: bend radius. If None, defaults to cross_section.radius.
        route_width: width of the route in um. If None, defaults to cross_section.width.
        auto_taper: add auto tapers.
        on_error: what to do on error. If error, raises an error. If None ignores the error.
        layer_transitions: dictionary of layer transitions to use for the routing when auto_taper=True.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component()
        mmi1 = c << gf.components.mmi1x2()
        mmi2 = c << gf.components.mmi1x2()
        mmi2.move((40, 20))
        gf.routing.route_single(c, mmi1.ports["o2"], mmi2.ports["o1"], radius=5, cross_section="strip")
        c.plot()
    """
    if cross_section is None and (layer is None or route_width is None):
        raise ValueError(
            f"Either {cross_section=} or {layer=} and route_width must be provided"
        )

    c = component
    p1 = port1
    p2 = port2
    port_type = port_type or p1.port_type

    if cross_section is None:
        cross_section = gf.cross_section.cross_section(
            layer=cast("LayerSpec", layer),
            width=cast("float", route_width),
            port_names=("e1", "e2") if port_type == "electrical" else ("o1", "o2"),
            port_types=(port_type, port_type),
        )

    if route_width:
        xs = gf.get_cross_section(cross_section, width=route_width)
    else:
        xs = gf.get_cross_section(cross_section)
    width = route_width or xs.width

    radius = radius or xs.radius
    bend90 = gf.get_component(bend, cross_section=xs, radius=radius, width=width)
    if auto_taper:
        p1 = add_auto_tapers(component, [p1], xs, layer_transitions)[0]
        p2 = add_auto_tapers(component, [p2], xs, layer_transitions)[0]

    def straight_dbu(width: int, length: int) -> kf.KCell:
        return gf.get_component(
            straight,
            length=c.kcl.to_um(length),
            cross_section=xs,
        ).to_itype()

    end_straight = c.kcl.to_dbu(end_straight_length)
    start_straight = c.kcl.to_dbu(start_straight_length)
    route_width = c.kcl.to_dbu(width)

    if steps and waypoints:
        raise ValueError("Provide either steps or waypoints, not both")

    waypoints_list = [] if waypoints is None else list(waypoints)

    if steps:
        x, y = p1.center
        for d in steps:
            if not STEP_DIRECTIVES.issuperset(d):
                raise ValueError(
                    f"Invalid step directives: {list(d.keys() - STEP_DIRECTIVES)}."
                    f"Valid directives are {list(STEP_DIRECTIVES)}"
                )
            x = d.get("x", x) + d.get("dx", 0)
            y = d.get("y", y) + d.get("dy", 0)
            waypoints_list.append((x, y))

    if waypoints_list:
        w: list[kf.kdb.Point] = []
        if not isinstance(waypoints_list[0], kf.kdb.DPoint):
            w.append(c.kcl.to_dbu(kf.kdb.DPoint(*p1.center)))
            for p in waypoints_list:
                if isinstance(p, tuple):
                    w.append(c.kcl.to_dbu(kf.kdb.DPoint(p[0], p[1])))
                else:
                    w.append(p.to_itype(c.kcl.dbu))
            w.append(c.kcl.to_dbu(kf.kdb.DPoint(*p2.center)))
        else:
            w = [
                p.to_itype(c.kcl.dbu)
                for p in cast("Sequence[gf.kdb.DPoint]", waypoints_list)
            ]

        try:
            return place_manhattan(
                component.to_itype(),
                p1=p1.to_itype(),
                p2=p2.to_itype(),
                straight_factory=straight_dbu,
                bend90_cell=bend90.to_itype(),
                pts=w,
                port_type=port_type,
                allow_width_mismatch=allow_width_mismatch,
                route_width=route_width,
            )
        except Exception as e:
            # error_route((ps, pe, router.start.pts, router.width))
            ps = p1
            pe = p2
            c = component
            pts = w
            db = kf.rdb.ReportDatabase("Route Placing Errors")
            cell = db.create_cell(
                (
                    c.kcl.future_cell_name or c.name
                    if c.name is not None and c.name.startswith("Unnamed_")
                    else c.name
                )
                or ""
            )
            cat = db.create_category(f"{ps.name} - {pe.name}")
            it = db.create_item(cell=cell, category=cat)
            it.add_value(
                f"Error while trying to place route from {ps.name} to {pe.name} at"
                f" points (dbu): {pts}"
            )
            it.add_value(f"Exception: {e}")
            path = kf.kdb.Path(pts, route_width or c.kcl.to_dbu(ps.width))
            it.add_value(c.kcl.to_um(path.polygon()))
            if on_error == "error":
                c.name = (
                    c.kcl.future_cell_name or c.name
                    if c.name is not None and c.name.startswith("Unnamed_")
                    else c.name
                ) or ""
                c.show(lyrdb=db)
                raise kf.routing.generic.PlacerError(
                    f"Error while trying to place route from {ps.name} to {pe.name} at"
                    f" points (dbu): {pts}"
                ) from e
            layer_error = (1, 0)
            layer_index = c.kcl.layer(*layer_error)
            c.shapes(layer_index).insert(path)
            return ManhattanRoute(
                backbone=pts,
                start_port=p1.to_itype(),
                end_port=p2.to_itype(),
            )

    else:
        return route(
            component.to_itype(),
            p1=p1.to_itype(),
            p2=p2.to_itype(),
            straight_factory=straight_dbu,
            bend90_cell=bend90.to_itype(),
            start_straight=start_straight,
            end_straight=end_straight,
            port_type=port_type,
            allow_width_mismatch=allow_width_mismatch,
            route_width=route_width,
        )


# FIXME
# route_single_electrical = partial(
#     route_single,
#     cross_section="metal_routing",
#     allow_width_mismatch=True,
#     port_type="electrical",
#     bend=wire_corner,
#     taper=None,
# )


def route_single_electrical(
    component: Component,
    port1: Port,
    port2: Port,
    start_straight_length: float | None = None,
    end_straight_length: float | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
) -> None:
    """Places a route between two electrical ports.

    Args:
        component: The cell to place the route in.
        port1: The first port.
        port2: The second port.
        start_straight_length: The length of the straight at the start of the route.
        end_straight_length: The length of the straight at the end of the route.
        layer: The layer of the route.
        width: The width of the route.
        cross_section: The cross section of the route.

    """
    c = component
    xs = gf.get_cross_section(cross_section)
    layer = layer or xs.layer
    width = width or xs.width
    layer = gf.get_layer(layer)
    start_straight_length = (
        c.kcl.to_dbu(start_straight_length) if start_straight_length else None
    )
    end_straight_length = (
        c.kcl.to_dbu(end_straight_length) if end_straight_length else None
    )
    route_elec(
        c=component.to_itype(),
        p1=port1.to_itype(),
        p2=port2.to_itype(),
        layer=layer,
        width=c.kcl.to_dbu(width),
        start_straight=start_straight_length,
        end_straight=end_straight_length,
    )

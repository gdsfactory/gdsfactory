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

from collections.abc import Mapping, Sequence
from typing import Literal, cast

import kfactory as kf
from kfactory.routing.electrical import route_elec
from kfactory.routing.generic import ManhattanRoute
from kfactory.routing.optical import place90, route

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.routing.auto_taper import add_auto_tapers
from gdsfactory.typings import (
    STEP_DIRECTIVES,
    ComponentSpec,
    CrossSectionSpec,
    LayerSpec,
    Port,
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
    steps: Sequence[Mapping[Literal["x", "y", "dx", "dy"], int | float]] | None = None,
    port_type: str | None = None,
    allow_width_mismatch: bool = False,
    radius: float | None = None,
    route_width: float | None = None,
    auto_taper: bool = True,
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

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component()
        mmi1 = c << gf.components.mmi1x2()
        mmi2 = c << gf.components.mmi1x2()
        mmi2.dmove((40, 20))
        gf.routing.route_single(c, mmi1.ports["o2"], mmi2.ports["o1"], radius=5, cross_section="strip")
        c.plot()
    """
    p1 = port1
    p2 = port2
    c = component

    if cross_section is None:
        if layer is None or route_width is None:
            raise ValueError(
                f"Either {cross_section=} or {layer=} and route_width must be provided"
            )

        elif radius:
            cross_section = gf.cross_section.cross_section(
                layer=layer,
                width=route_width,
                radius=radius,
            )
        else:
            cross_section = gf.cross_section.cross_section(
                layer=layer,
                width=route_width,
            )

    port_type = port_type or p1.port_type
    xs = gf.get_cross_section(cross_section)
    width = route_width or xs.width
    radius = radius or xs.radius

    bend90 = gf.get_component(bend, cross_section=cross_section, radius=radius)
    if auto_taper:
        p1 = add_auto_tapers(component, [p1], cross_section)[0]
        p2 = add_auto_tapers(component, [p2], cross_section)[0]

    def straight_dbu(width: int, length: int) -> kf.KCell:
        return gf.get_component(
            straight,
            length=c.kcl.to_um(length),
            cross_section=cross_section,
        ).to_itype()

    end_straight = c.kcl.to_dbu(end_straight_length)
    start_straight = c.kcl.to_dbu(start_straight_length)
    route_width = c.kcl.to_dbu(width)

    if steps and waypoints:
        raise ValueError("Provide either steps or waypoints, not both")

    waypoints_list = [] if waypoints is None else list(waypoints)
    if steps is None:
        steps = []

    if steps:
        x, y = port1.center
        for d in steps:
            if not STEP_DIRECTIVES.issuperset(d):
                invalid_step_directives = list(set[str](d.keys()) - STEP_DIRECTIVES)
                raise ValueError(
                    f"Invalid step directives: {invalid_step_directives}."
                    f"Valid directives are {list(STEP_DIRECTIVES)}"
                )
            x = float(d.get("x", x) + d.get("dx", 0.0))
            y = float(d.get("y", y) + d.get("dy", 0.0))
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
                for p in cast(Sequence[gf.kdb.DPoint], waypoints_list)
            ]

        try:
            return place90(
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
                c.kcl.future_cell_name or c.name
                if c.name.startswith("Unnamed_")
                else c.name
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
            c.name = (
                c.kcl.future_cell_name or c.name
                if c.name.startswith("Unnamed_")
                else c.name
            )
            c.show(lyrdb=db)
            raise kf.routing.generic.PlacerError(
                f"Error while trying to place route from {ps.name} to {pe.name} at"
                f" points (dbu): {pts}"
            ) from e

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


if __name__ == "__main__":
    # c = gf.Component(name="demo")
    # s = gf.c.wire_straight()
    # pt = c << s
    # pb = c << s
    # pt.dmove((50, 50))
    # gf.routing.route_single_electrical(
    #     c,
    #     pb.ports["e2"],
    #     pt.ports["e1"],
    #     cross_section="strip",
    #     start_straight_length=10,
    #     end_straight_length=30,
    # )
    # c.show()

    # c = gf.Component(name="waypoints_sample")
    # w = gf.components.straight()
    # left = c << w
    # right = c << w
    # right.dmove((100, 80))

    # obstacle = gf.components.rectangle(size=(100, 10))
    # obstacle1 = c << obstacle
    # obstacle2 = c << obstacle
    # obstacle1.dymin = 40
    # obstacle2.dxmin = 25

    # p0 = left.ports["o2"]
    # p1 = right.ports["o2"]
    # p0x, p0y = left.ports["o2"].center
    # p1x, p1y = right.ports["o2"].center
    # o = 10  # vertical offset to overcome bottom obstacle
    # ytop = 20

    # r = gf.routing.route_single(
    #     c,
    #     p0,
    #     p1,
    #     cross_section="rib",
    #     waypoints=[
    #         (p0x + o, p0y),
    #         (p0x + o, ytop),
    #         (p1x + o, ytop),
    #         (p1x + o, p1y),
    #     ],
    # )
    # c.show()

    # c = gf.Component(name="electrical")
    # w = gf.components.wire_straight()
    # left = c << w
    # right = c << w
    # right.dmove((100, 80))
    # obstacle = gf.components.rectangle(size=(100, 10))
    # obstacle1 = c << obstacle
    # obstacle2 = c << obstacle
    # obstacle1.dymin = 40
    # obstacle2.dxmin = 25

    # p0 = left.ports["e2"]
    # p1 = right.ports["e2"]
    # p0x, p0y = left.ports["e2"].center
    # p1x, p1y = right.ports["e2"].center
    # o = 10  # vertical offset to overcome bottom obstacle
    # ytop = 20

    # r = route_single(
    #     c,
    #     p0,
    #     p1,
    #     cross_section="metal_routing",
    #     waypoints=[
    #         (p0x + o, p0y),
    #         (p0x + o, ytop),
    #         (p1x + o, ytop),
    #         (p1x + o, p1y),
    #     ],
    # )
    # c.show()

    # c = gf.Component()
    # top = c << gf.components.straight(
    #     length=0.1, cross_section="metal_routing", width=40
    # )
    # bot = c << gf.components.straight(
    #     length=0.1, cross_section="metal_routing", width=20
    # )
    # d = 200
    # bot.dmove((d, d))

    # p0 = top.ports["e2"]
    # p1 = bot.ports["e1"]
    # r = gf.routing.route_single(c, p0, p1, cross_section="metal_routing")
    # c.show()
    # import gdsfactory as gf

    # w = gf.components.straight()
    # left = c << w
    # right = c << w
    # right.dmove((500, 80))

    # obstacle = gf.components.rectangle(size=(100, 10), port_type=None)
    # obstacle1 = c << obstacle
    # obstacle2 = c << obstacle
    # obstacle1.dymin = 40
    # obstacle2.dxmin = 25

    # p1 = left.ports["o2"]
    # p2 = right.ports["o2"]
    # route_single(
    #     c,
    #     port1=p1,
    #     port2=p2,
    #     # steps=[
    #     #     {"x": 20},
    #     #     {"y": 20},
    #     #     {"x": 120},
    #     #     {"y": 80},
    #     # ],
    #     cross_section="strip",
    #     # layer=(2, 0),
    #     route_width=0.9,
    # )

    # c = gf.Component()
    # mmi1 = c << gf.components.mmi1x2()
    # mmi2 = c << gf.components.mmi1x2()
    # mmi2.dmove((100, 50))
    # route = gf.routing.route_single(
    #     c,
    #     port1=mmi1.ports["o2"],
    #     port2=mmi2.ports["o1"],
    #     cross_section="rib",  # layer=(1, 0), route_width=2
    # )
    # c.show()

    c = gf.Component()
    s1 = c << gf.components.straight()
    s2 = c << gf.components.straight(width=2)
    s2.dmove((100, 50))
    route_ = gf.routing.route_single(
        c,
        port1=s1.ports["o2"],
        port2=s2.ports["o1"],
        cross_section="strip",
        auto_taper=True,
    )
    c.show()

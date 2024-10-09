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
from typing import Literal

import kfactory as kf
from kfactory.routing.electrical import route_elec
from kfactory.routing.generic import ManhattanRoute
from kfactory.routing.optical import place90, route

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.port import Port
from gdsfactory.routing.auto_taper import add_auto_tapers
from gdsfactory.typings import (
    STEP_DIRECTIVES,
    ComponentSpec,
    Coordinates,
    CrossSectionSpec,
    LayerSpec,
    MultiCrossSectionAngleSpec,
)


def route_single(
    component: Component,
    port1: Port,
    port2: Port,
    bend: ComponentSpec = bend_euler,
    straight: ComponentSpec = straight_function,
    start_straight_length: float = 0.0,
    end_straight_length: float = 0.0,
    cross_section: CrossSectionSpec | MultiCrossSectionAngleSpec = "strip",
    waypoints: Coordinates | None = None,
    steps: Sequence[Mapping[Literal["x", "y", "dx", "dy"], int | float]] | None = None,
    port_type: str | None = None,
    allow_width_mismatch: bool = False,
    radius: float | None = None,
    route_width: float | None = None,
    auto_taper: bool = True,
) -> ManhattanRoute:
    """Returns a Manhattan Route between 2 ports.

    The references are straights, bends and tapers.
    `route_single` is an automatic version of `route_single_from_steps`.

    Args:
        component: to place the route into.
        port1: start port.
        port2: end port.
        bend: bend spec.
        straight: straight spec.
        start_straight_length: length of starting straight.
        end_straight_length: length of end straight.
        cross_section: spec.
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

        c = gf.Component('sample_connect')
        mmi1 = c << gf.components.mmi1x2()
        mmi2 = c << gf.components.mmi1x2()
        mmi2.dmove((40, 20))
        gf.routing.route_single(c, mmi1.ports["o2"], mmi2.ports["o1"], radius=5)
        c.plot()
    """
    p1 = port1
    p2 = port2

    port_type = port_type or p1.port_type
    if route_width:
        xs = gf.get_cross_section(cross_section, width=route_width)
    else:
        xs = gf.get_cross_section(cross_section)
    width = route_width or xs.width
    radius = radius or xs.radius
    width_dbu = width / component.kcl.dbu

    bend90 = gf.get_component(
        bend, cross_section=cross_section, radius=radius, width=width
    )
    if auto_taper:
        p1 = add_auto_tapers(component, [p1], cross_section)[0]
        p2 = add_auto_tapers(component, [p2], cross_section)[0]

    def straight_dbu(
        length: int,
        width: int = width_dbu,
        cross_section=cross_section,
    ) -> Component:
        return gf.get_component(
            straight,
            length=length * component.kcl.dbu,
            width=width * component.kcl.dbu,
            cross_section=cross_section,
        )

    dbu = component.kcl.dbu
    end_straight = round(end_straight_length / dbu)
    start_straight = round(start_straight_length / dbu)
    route_width = round(width / dbu)

    if steps and waypoints:
        raise ValueError("Provide either steps or waypoints, not both")

    if waypoints is None:
        waypoints = []

    if steps is None:
        steps = []

    if steps:
        x, y = port1.dcenter
        for d in steps:
            if not STEP_DIRECTIVES.issuperset(d):
                invalid_step_directives = list(set(d.keys()) - STEP_DIRECTIVES)
                raise ValueError(
                    f"Invalid step directives: {invalid_step_directives}."
                    f"Valid directives are {list(STEP_DIRECTIVES)}"
                )
            x = d.get("x", x) + d.get("dx", 0)
            y = d.get("y", y) + d.get("dy", 0)
            waypoints += [(x, y)]

    if len(waypoints) > 0:
        if not isinstance(waypoints[0], kf.kdb.Point):
            w = [kf.kdb.Point(*p1.center)]
            w += [kf.kdb.Point(p[0] / dbu, p[1] / dbu) for p in waypoints]
            w += [kf.kdb.Point(*p2.center)]
            waypoints = w

        return place90(
            component,
            p1=p1,
            p2=p2,
            straight_factory=straight_dbu,
            bend90_cell=bend90,
            pts=waypoints,
            port_type=port_type,
            allow_width_mismatch=allow_width_mismatch,
            route_width=route_width,
        )

    else:
        return route(
            component,
            p1=p1,
            p2=p2,
            straight_factory=straight_dbu,
            bend90_cell=bend90,
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
    cross_section: CrossSectionSpec = "metal3",
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
    xs = gf.get_cross_section(cross_section)
    layer = layer or xs.layer
    width = width or xs.width
    layer = gf.get_layer(layer)
    start_straight_length = (
        start_straight_length / component.kcl.dbu if start_straight_length else None
    )
    end_straight_length = (
        end_straight_length / component.kcl.dbu if end_straight_length else None
    )
    route_elec(
        c=component,
        p1=port1,
        p2=port2,
        layer=layer,
        width=round(width / component.kcl.dbu),
        start_straight=start_straight_length,
        end_straight=end_straight_length,
    )


if __name__ == "__main__":
    # c = gf.Component("demo")
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

    # c = gf.Component("waypoints_sample")
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
    # p0x, p0y = left.ports["o2"].dcenter
    # p1x, p1y = right.ports["o2"].dcenter
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

    # c = gf.Component("electrical")
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
    # p0x, p0y = left.ports["e2"].dcenter
    # p1x, p1y = right.ports["e2"].dcenter
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
    import gdsfactory as gf

    c = gf.Component("route_single_from_steps_sample")
    w = gf.components.straight()
    left = c << w
    right = c << w
    right.dmove((500, 80))

    obstacle = gf.components.rectangle(size=(100, 10), port_type=None)
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle1.dymin = 40
    obstacle2.dxmin = 25

    p1 = left.ports["o2"]
    p2 = right.ports["o2"]
    route_single(
        c,
        port1=p1,
        port2=p2,
        steps=[
            {"x": 20},
            {"y": 20},
            {"x": 120},
            {"y": 80},
        ],
    )
    c.show()

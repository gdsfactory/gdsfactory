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

import kfactory as kf
from kfactory.routing.electrical import route_elec
from kfactory.routing.optical import OpticalManhattanRoute, place90, route

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.port import Port
from gdsfactory.typings import (
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
    taper: ComponentSpec | None = None,
    start_straight_length: float = 0.0,
    end_straight_length: float = 0.0,
    cross_section: CrossSectionSpec | MultiCrossSectionAngleSpec = "strip",
    waypoints: Coordinates | None = None,
    port_type: str | None = None,
    allow_width_mismatch: bool = False,
    radius: float | None = None,
    route_width: float | None = None,
) -> OpticalManhattanRoute:
    """Returns a Manhattan Route between 2 ports.

    The references are straights, bends and tapers.
    `route_single` is an automatic version of `route_single_from_steps`.

    Args:
        component: to place the route into.
        port1: start port.
        port2: end port.
        bend: bend spec.
        straight: straight spec.
        taper: taper spec.
        start_straight_length: length of starting straight.
        end_straight_length: length of end straight.
        cross_section: spec.
        waypoints: list of points to pass through.
        port_type: port type to route.
        allow_width_mismatch: allow different port widths.
        radius: bend radius. If None, defaults to cross_section.radius.
        route_width: width of the route. If None, defaults to cross_section.width.


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
    xs = gf.get_cross_section(cross_section)
    width = xs.width
    radius = radius or xs.radius
    width_dbu = width / component.kcl.dbu

    taper_cell = gf.get_component(taper, cross_section=cross_section) if taper else None
    bend90 = (
        bend
        if isinstance(bend, Component)
        else gf.get_component(bend, cross_section=cross_section, radius=radius)
    )

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

    route_width = route_width or round(xs.width / dbu)

    if waypoints is not None:
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
            taper_cell=taper_cell,
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
            taper_cell=taper_cell,
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

    # r = route_single_electrical(
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
    c = gf.Component()
    w = gf.components.straight()
    top = c << w
    bot = c << w
    bot.dmove((0, -2))

    p0 = top.ports["o2"]
    p1 = bot.ports["o2"]

    r = gf.routing.route_single(c, p0, p1, cross_section="rib", taper=None)
    c.show()

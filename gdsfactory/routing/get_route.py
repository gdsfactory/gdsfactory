"""`get_route` returns a Manhattan route between two ports.

`get_route` only works for an individual routes. For routing groups of ports you need to use `get_bundle` instead

To make a route, you need to supply:

 - input port
 - output port
 - bend
 - straight
 - taper to taper to wider straights and reduce straight loss (Optional)


To generate a straight route:

 1. Generate the backbone of the route.
 This is a list of manhattan coordinates that the route would pass through
 if it used only sharp bends (right angles)

 2. Replace the corners by bend references
 (with rotation and position computed from the manhattan backbone)

 3. Add tapers if needed and if space permits

 4. generate straight portions in between tapers or bends


 A `Route` is a dataclass with:

- references: list of references for tapers, bends and straight waveguides
- ports: a dict of port name to Port, usually two ports "input" and "output"
- length: a float with the length of the route

"""
from __future__ import annotations

import warnings
from functools import partial

import kfactory as kf
from kfactory.routing.optical import OpticalManhattanRoute, place90, route

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.components.via_corner import via_corner
from gdsfactory.components.wire import wire_corner
from gdsfactory.cross_section import metal2, metal3
from gdsfactory.port import Port
from gdsfactory.typings import (
    ComponentSpec,
    Coordinates,
    CrossSectionSpec,
    MultiCrossSectionAngleSpec,
)


def get_route(*args, **kwargs) -> None:
    raise ValueError(
        "get_route is not supported in gdsfactory>=8. Use place_route instead!"
    )


get_route_from_waypoints = get_route


get_route_electrical = partial(
    get_route,
    bend=wire_corner,
    cross_section="xs_metal_routing",
    taper=None,
)

get_route_electrical_m2 = partial(
    get_route,
    bend=wire_corner,
    cross_section=metal2,
    taper=None,
)

get_route_electrical_multilayer = partial(
    get_route_electrical,
    bend=via_corner,
    cross_section=[(metal2, (0, 180)), (metal3, (90, 270))],
)

get_route_from_waypoints_electrical = partial(
    get_route_from_waypoints, bend=wire_corner, cross_section="xs_metal_routing"
)

get_route_from_waypoints_electrical_m2 = partial(
    get_route_from_waypoints, bend=wire_corner, cross_section=metal2
)

get_route_from_waypoints_electrical_multilayer = partial(
    get_route_from_waypoints,
    bend=via_corner,
    cross_section=[(metal2, (0, 180)), (metal3, (90, 270))],
)


def place_route(
    component: Component,
    port1: Port,
    port2: Port,
    bend: ComponentSpec = bend_euler,
    straight: ComponentSpec = straight_function,
    taper: ComponentSpec | None = taper_function,
    start_straight_length: float = 0.0,
    end_straight_length: float = 0.0,
    cross_section: CrossSectionSpec | MultiCrossSectionAngleSpec = "xs_sc",
    waypoints: Coordinates | None = None,
    **kwargs,
) -> OpticalManhattanRoute:
    """Returns a Manhattan Route between 2 ports.

    The references are straights, bends and tapers.
    `place_route` is an automatic version of `place_route_from_steps`.

    Args:
        component: to place the route into.
        input_port: start port.
        output_port: end port.
        bend: bend spec.
        straight: straight spec.
        taper: taper spec.
        start_straight_length: length of starting straight.
        end_straight_length: length of end straight.
        cross_section: spec.
        kwargs: cross_section settings.


    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component('sample_connect')
        mmi1 = c << gf.components.mmi1x2()
        mmi2 = c << gf.components.mmi1x2()
        mmi2.move((40, 20))
        gf.routing.place_route(c, mmi1.ports["o2"], mmi2.ports["o1"], radius=5)
        c.plot()
    """
    p1 = port1
    p2 = port2
    with_sbend = kwargs.pop("min_straight_length", None)
    min_straight_length = kwargs.pop("min_straight_length", None)

    if with_sbend:
        warnings.warn("with_sbend is not implemented yet")

    if min_straight_length:
        warnings.warn("minimum straight length not implemented yet")

    xs = gf.get_cross_section(cross_section, **kwargs)
    width = xs.width
    width_dbu = width / component.kcl.dbu
    straight = partial(straight, width=width)
    bend90 = gf.get_component(bend_euler)
    taper_cell = gf.get_component(taper) if taper else None

    def straight_dbu(
        length: int, width: int = width_dbu, cross_section=cross_section, **kwargs
    ) -> Component:
        return straight(
            length=round(length * component.kcl.dbu),
            width=round(width * component.kcl.dbu),
            cross_section=cross_section,
            **kwargs,
        )

    dbu = component.kcl.dbu
    end_straight = round(end_straight_length / dbu)
    start_straight = round(start_straight_length / dbu)

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
        )

    return route(
        component,
        p1=p1,
        p2=p2,
        straight_factory=straight_dbu,
        bend90_cell=bend90,
        taper_cell=taper_cell,
        start_straight=start_straight,
        end_straight=end_straight,
    )


if __name__ == "__main__":
    # c = gf.Component("demo")
    # s = gf.c.straight()
    # pt = c << s
    # pb = c << s
    # pt.d.move((50, 50))
    # gf.routing.place_route(
    #     c,
    #     pb.ports["o2"],
    #     pt.ports["o1"],
    #     cross_section="xs_sc_auto_widen",
    # )
    # c.show()
    c = gf.Component("waypoints_sample")

    w = gf.components.straight()
    left = c << w
    right = c << w
    right.d.move((100, 80))

    obstacle = gf.components.rectangle(size=(100, 10))
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle1.d.ymin = 40
    obstacle2.d.xmin = 25

    p0 = left.ports["o2"]
    p1 = right.ports["o2"]
    p0x, p0y = left.ports["o2"].d.center
    p1x, p1y = right.ports["o2"].d.center
    o = 10  # vertical offset to overcome bottom obstacle
    ytop = 20

    routes = gf.routing.place_route(
        c,
        p0,
        p1,
        waypoints=[
            (p0x + o, p0y),
            (p0x + o, ytop),
            (p1x + o, ytop),
            (p1x + o, p1y),
        ],
    )
    c.show()

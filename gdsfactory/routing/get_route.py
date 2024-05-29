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

from collections.abc import Callable
from functools import partial

import numpy as np
from pydantic import validate_call

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.components.via_corner import via_corner
from gdsfactory.components.wire import wire_corner
from gdsfactory.cross_section import metal2, metal3
from gdsfactory.port import Port
from gdsfactory.routing.manhattan import round_corners, route_manhattan
from gdsfactory.typings import (
    ComponentSpec,
    Coordinates,
    CrossSectionSpec,
    MultiCrossSectionAngleSpec,
    Route,
)


@validate_call
def get_route(
    input_port: Port,
    output_port: Port,
    bend: ComponentSpec = bend_euler,
    with_sbend: bool = False,
    straight: ComponentSpec = straight_function,
    taper: ComponentSpec | None = None,
    start_straight_length: float | None = None,
    end_straight_length: float | None = None,
    min_straight_length: float = 10e-3,
    auto_widen: bool = False,
    auto_widen_minimum_length: float = 100,
    taper_length: float = 10,
    width_wide: float = 2,
    cross_section: None | CrossSectionSpec | MultiCrossSectionAngleSpec = "xs_sc",
    **kwargs,
) -> Route:
    """Returns a Manhattan Route between 2 ports.

    The references are straights, bends and tapers.
    `get_route` is an automatic version of `get_route_from_steps`.

    Args:
        input_port: start port.
        output_port: end port.
        bend: bend spec.
        with_sbend: add sbend in case there are routing errors.
        straight: straight spec.
        taper: taper spec.
        start_straight_length: length of starting straight.
        end_straight_length: length of end straight.
        min_straight_length: min length of straight for any intermediate segment.
        auto_widen: auto widen the straights.
        auto_widen_minimum_length: minimum length to auto widen.
        taper_length: length of taper.
        width_wide: width of the wider straight.
        cross_section: spec.
        kwargs: cross_section settings.


    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component('sample_connect')
        mmi1 = c << gf.components.mmi1x2()
        mmi2 = c << gf.components.mmi1x2()
        mmi2.move((40, 20))
        route = gf.routing.get_route(mmi1.ports["o2"], mmi2.ports["o1"], radius=5)
        c.add(route.references)
        c.plot()

    """

    if isinstance(cross_section, list | tuple):
        xs_list = []
        for element in cross_section:
            xs, angles = element
            xs = gf.get_cross_section(xs, **kwargs)
            xs_list.append((xs, angles))
        cross_section = xs_list

    else:
        cross_section = gf.get_cross_section(cross_section, **kwargs)

    if cross_section:
        bend90 = (
            bend
            if isinstance(bend, Component)
            else gf.get_component(bend, cross_section=cross_section)
        )
    else:
        bend90 = gf.get_component(bend)
    if taper and cross_section:
        if isinstance(cross_section, tuple | list):
            raise ValueError(
                "Tapers not implemented for routes made from multiple cross_sections."
            )
        width1 = input_port.width
        width2 = width_wide if auto_widen else width1

        taper = gf.get_component(
            taper,
            length=taper_length,
            width1=input_port.width,
            width2=width2,
            cross_section=cross_section,
        )

    elif taper and cross_section is None:
        taper = gf.get_component(taper)

    return route_manhattan(
        input_port=input_port,
        output_port=output_port,
        straight=straight,
        taper=taper,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        min_straight_length=min_straight_length,
        bend=bend90,
        with_sbend=with_sbend,
        cross_section=cross_section,
        auto_widen=auto_widen,
        auto_widen_minimum_length=auto_widen_minimum_length,
        width_wide=width_wide,
        taper_length=taper_length,
    )


def get_route_electrical(
    input_port: Port,
    output_port: Port,
    bend: ComponentSpec = wire_corner,
    straight: ComponentSpec = straight_function,
    start_straight_length: float | None = None,
    end_straight_length: float | None = None,
    min_straight_length: float | None = None,
    gap: float = 10,
    cross_section: None
    | CrossSectionSpec
    | MultiCrossSectionAngleSpec = "xs_metal_routing",
    **kwargs,
) -> Route:
    """Returns a Manhattan Route between 2 ports with electrical routing.

    Args:
        input_port: start port.
        output_port: end port.
        bend: bend spec.
        straight: straight spec.
        start_straight_length: length of starting straight.
        end_straight_length: length of end straight.
        cross_section: spec.
        kwargs: cross_section settings.
    """

    if isinstance(cross_section, list | tuple):
        xs_list = []
        for element in cross_section:
            xs, angles = element
            xs = gf.get_cross_section(xs, **kwargs)
            xs_list.append((xs, angles))
        cross_section = xs_list
    else:
        xs = gf.get_cross_section(cross_section, **kwargs)

    min_straight_length = min_straight_length or xs.width + gap
    start_straight_length = start_straight_length or min_straight_length
    end_straight_length = end_straight_length or min_straight_length

    return get_route(
        input_port=input_port,
        output_port=output_port,
        bend=bend,
        straight=straight,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        min_straight_length=min_straight_length,
        cross_section=cross_section,
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


def get_route_from_waypoints(
    waypoints: Coordinates,
    bend: Callable = bend_euler,
    straight: Callable = straight_function,
    taper: Callable | None = taper_function,
    cross_section: CrossSectionSpec | None = "xs_sc",
    auto_widen: bool = False,
    auto_widen_minimum_length: float = 100,
    taper_length: float = 10,
    width_wide: float = 2,
    **kwargs,
) -> Route:
    """Returns a route formed by the given waypoints with bends instead of \
    corners and optionally tapers in straight sections. Tapering to wider \
    straights reduces the optical loss. `get_route_from_waypoints` is a manual \
    version of `get_route` `get_route_from_steps` is a  more concise and \
    convenient version of `get_route_from_waypoints` also available in \
    gf.routing.

    Args:
        waypoints: Coordinates that define the route.
        bend: function that returns bends.
        straight: function that returns straight waveguides.
        taper: function that returns tapers.
        cross_section: spec.
        auto_widen: auto widen the straights.
        taper_length: length of taper.
        width_wide: width of the wider straight.
        kwargs: cross_section settings.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component("waypoints_sample")

        w = gf.components.straight()
        left = c << w
        right = c << w
        right.move((100, 80))

        obstacle = gf.components.rectangle(size=(100, 10))
        obstacle1 = c << obstacle
        obstacle2 = c << obstacle
        obstacle1.ymin = 40
        obstacle2.xmin = 25


        p0x, p0y = left.ports["o2"].center
        p1x, p1y = right.ports["o2"].center
        o = 10  # vertical offset to overcome bottom obstacle
        ytop = 20


        routes = gf.routing.get_route_from_waypoints(
            [
                (p0x, p0y),
                (p0x + o, p0y),
                (p0x + o, ytop),
                (p1x + o, ytop),
                (p1x + o, p1y),
                (p1x, p1y),
            ],
        )
        c.add(routes.references)
        c.plot()

    """
    if isinstance(cross_section, list | tuple):
        xs_list = []
        for element in cross_section:
            xs, angles = element
            xs = gf.get_cross_section(xs, **kwargs)
            xs_list.append((xs, angles))
        x = xs_list
    elif cross_section:
        kwargs.pop("start_straight_length", None)
        kwargs.pop("end_straight_length", None)
        x = cross_section = gf.get_cross_section(cross_section, **kwargs)

    if isinstance(cross_section, list):
        taper = None
    elif cross_section and taper:
        x = gf.get_cross_section(cross_section, **kwargs)
        auto_widen = auto_widen
        width1 = x.width
        width2 = width_wide if auto_widen else width1
        taper_length = taper_length
        if auto_widen:
            taper = (
                taper(
                    length=taper_length,
                    width1=width1,
                    width2=width2,
                    cross_section=x,
                )
                if callable(taper)
                else taper
            )
        else:
            taper = None
    else:
        taper = None
        x = None
    waypoints = np.array(waypoints)
    kwargs.pop("route_filter", "")

    return round_corners(
        points=waypoints,
        bend=bend,
        straight=straight,
        taper=taper,
        cross_section=x,
        auto_widen=auto_widen,
        auto_widen_minimum_length=auto_widen_minimum_length,
        taper_length=taper_length,
        width_wide=width_wide,
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


if __name__ == "__main__":
    # w = gf.components.mmi1x2()
    # c = gf.Component()
    # c << w
    # route = get_route(w.ports["o2"], w.ports["o1"], layer=(2, 0), width=2)
    # cc = c.add(route.references)
    # cc.show(show_ports=True)

    # c = gf.Component("multi-layer")
    # ptop = c << gf.components.pad_array(orientation=270)
    # pbot = c << gf.components.pad_array(orientation=90)

    # ptop.movex(300)
    # ptop.movey(300)
    # route = get_route_electrical_multilayer(
    #     ptop.ports["e11"],
    #     pbot.ports["e11"],
    #     end_straight_length=100,
    # )
    # c.add(route.references)
    # c.show()

    c = gf.Component("two_pads")
    ptop = c << gf.components.pad(port_orientation=270)
    pbot = c << gf.components.pad(port_orientation=90)

    ptop.movex(300)
    ptop.movey(300)
    route = get_route(
        ptop.ports["pad"],
        pbot.ports["pad"],
        cross_section="xs_sc",
        bend=gf.c.wire_corner45,
        width=5,
    )
    c.add(route.references)
    c.show()

    # c = gf.Component("sample_connect")
    # mmi1 = c << gf.components.mmi1x2()
    # mmi2 = c << gf.components.mmi1x2()
    # mmi2.move((200, 50))

    # bend = partial(gf.components.bend_euler, cross_section="xs_rc")
    # straight = partial(gf.components.straight, cross_section="xs_rc")

    # via_along_path = gf.cross_section.ComponentAlongPath(component=gf.c.via1, spacing=1)
    # xs_with_vias = gf.cross_section.strip(components_along_path=[via_along_path])

    # route = gf.routing.get_route(
    #     mmi1.ports["o3"],
    #     mmi2.ports["o1"],
    #     bend=bend,
    #     straight=straight,
    #     auto_widen=True,
    #     width_wide=2,
    #     auto_widen_minimum_length=100,
    #     radius=30,
    #     # cross_section=None,
    #     cross_section=xs_with_vias,
    # )
    # c.add(route.references)
    # c.show()

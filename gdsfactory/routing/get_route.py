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
from functools import partial
from typing import Callable, Optional

import numpy as np

from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.components.wire import wire_corner
from gdsfactory.cross_section import metal3, strip
from gdsfactory.port import Port
from gdsfactory.routing.manhattan import round_corners, route_manhattan
from gdsfactory.types import (
    ComponentFactory,
    ComponentOrFactory,
    Coordinates,
    CrossSectionFactory,
    Route,
)


def get_route(
    input_port: Port,
    output_port: Port,
    bend: ComponentOrFactory = bend_euler,
    straight: ComponentOrFactory = straight_function,
    taper: Optional[ComponentFactory] = None,
    start_straight_length: float = 0.01,
    end_straight_length: float = 0.01,
    min_straight_length: float = 0.01,
    cross_section: CrossSectionFactory = strip,
    **kwargs,
) -> Route:
    """Returns a Manhattan Route between 2 ports
    The references are straights, bends and tapers.
    `get_route` is an automatic version of `get_route_from_steps`

    Args:
        input_port: start port
        output_port: end port
        bend: function that return bends
        straight: function that returns straights
        taper:
        start_straight_length: length of starting straight
        end_straight_length: length of end straight
        min_straight_length: min length of straight for any intermediate segment
        cross_section:
        kwargs: cross_section settings


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
    x = cross_section(**kwargs)
    taper_length = x.info.get("taper_length")
    width1 = input_port.width
    auto_widen = x.info.get("auto_widen", False)
    width2 = x.info.get("width_wide") if auto_widen else width1

    bend90 = bend(cross_section=cross_section, **kwargs) if callable(bend) else bend

    if taper:
        taper = partial(
            taper,
            length=taper_length,
            width1=input_port.width,
            width2=width2,
            cross_section=cross_section,
            **kwargs,
        )

    return route_manhattan(
        input_port=input_port,
        output_port=output_port,
        straight=straight,
        taper=taper,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        min_straight_length=min_straight_length,
        bend=bend90,
        cross_section=cross_section,
        **kwargs,
    )


get_route_electrical = partial(
    get_route,
    bend=wire_corner,
    start_straight_length=10,
    end_straight_length=10,
    cross_section=metal3,
    taper=None,
)


def get_route_from_waypoints(
    waypoints: Coordinates,
    bend: Callable = bend_euler,
    straight: Callable = straight_function,
    taper: Optional[Callable] = taper_function,
    cross_section: CrossSectionFactory = strip,
    **kwargs,
) -> Route:
    """Returns a route formed by the given waypoints with
    bends instead of corners and optionally tapers in straight sections.
    Tapering to wider straights reduces the optical loss.
    `get_route_from_waypoints` is a manual version of `get_route`
    `get_route_from_steps` is a  more concise and convenient version of
    `get_route_from_waypoints` also available in gf.routing

    Args:
        waypoints: Coordinates that define the route
        bend: function that returns bends
        straight: function that returns straight waveguides
        taper: function that returns tapers
        cross_section:
        kwargs: cross_section settings

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


        p0x, p0y = left.ports["o2"].midpoint
        p1x, p1y = right.ports["o2"].midpoint
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

    x = cross_section(**kwargs)
    auto_widen = x.info.get("auto_widen", False)
    width1 = x.info.get("width")
    width2 = x.info.get("width_wide") if auto_widen else width1
    taper_length = x.info.get("taper_length")
    waypoints = np.array(waypoints)
    kwargs.pop("route_filter", "")

    if auto_widen:
        taper = (
            taper(
                length=taper_length,
                width1=width1,
                width2=width2,
                cross_section=cross_section,
                **kwargs,
            )
            if callable(taper)
            else taper
        )
    else:
        taper = None

    return round_corners(
        points=waypoints,
        bend=bend,
        straight=straight,
        taper=taper,
        cross_section=cross_section,
        **kwargs,
    )


if __name__ == "__main__":
    import gdsfactory as gf

    # w = gf.components.mmi1x2()
    # c = gf.Component()
    # c << w
    # route = get_route(w.ports["o2"], w.ports["o1"], layer=(2, 0), width=2)
    # cc = c.add(route.references)
    # cc.show()

    c = gf.Component()
    p1 = c << gf.c.pad_array270()
    p2 = c << gf.c.pad_array90()

    p1.movex(300)
    p1.movey(300)
    route = get_route_electrical(
        p1.ports["e13"],
        p2.ports["e11"],
        cross_section=gf.cross_section.metal3,
        width=10.0,
    )
    c.add(route.references)

    c.show()

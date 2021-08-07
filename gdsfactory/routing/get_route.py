"""`get_route` returns a Manhattan route between two ports.

`get_route` only works for an individual routes. For routing groups of ports you need to use `get_bundle` instead

To make a route, you need to supply:

 - input port
 - output port
 - bend_factory
 - straight_factory
 - taper_factory to taper to wider straights and reduce straight loss (Optional)


To generate a straight route:

 1. Generate the backbone of the route.
 This is a list of manhattan coordinates that the route would pass through
 if it used only sharp bends (right angles)

 2. Replace the corners by bend references (with rotation and position computed from the manhattan backbone)

 3. Add tapers if needed and if space permits

 4. generate straight portions in between tapers or bends


 A `Route` is a dataclass with:

- references: list of references for tapers, bends and straight waveguides
- ports: a dict of port name to Port, usually two ports "input" and "output"
- length: a float with the length of the route

"""

from typing import Callable, Optional

import numpy as np

from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.cross_section import (
    StrOrDict,
    get_cross_section,
    get_waveguide_settings,
)
from gdsfactory.port import Port
from gdsfactory.routing.manhattan import round_corners, route_manhattan
from gdsfactory.types import ComponentOrFactory, Coordinates, Number, Route


def get_route(
    input_port: Port,
    output_port: Port,
    bend_factory: ComponentOrFactory = bend_euler,
    straight_factory: ComponentOrFactory = straight,
    taper_factory: Optional[ComponentOrFactory] = taper_function,
    start_straight: Number = 0.01,
    end_straight: Number = 0.01,
    min_straight: Number = 0.01,
    waveguide: str = "strip",
    **waveguide_settings,
) -> Route:
    """Returns a Manhattan Route between 2 ports
    The references are straights, bends and tapers.
    `get_route` is an automatic version of `get_route_from_steps`

    Args:
        input_port: start port
        output_port: end port
        bend_factory: function that return bends
        straight_factory: function that returns straights
        taper_factory:
        start_straight: length of starting straight
        end_straight: Number: length of end straight
        min_straight: Number: min length of straight
        waveguide: waveguide definition from TECH.waveguide
        waveguide_settings:


    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component('sample_connect')
        mmi1 = c << gf.components.mmi1x2()
        mmi2 = c << gf.components.mmi1x2()
        mmi2.move((100, 50))
        route = gf.routing.get_route(mmi1.ports["E1"], mmi2.ports["W0"])
        c.add(route.references)
        c.show()

    """
    waveguide_settings = get_waveguide_settings(waveguide, **waveguide_settings)
    taper_length = waveguide_settings.get("taper_length")
    width1 = input_port.width
    auto_widen = waveguide_settings.get("auto_widen", False)
    width2 = waveguide_settings.get("width_wide") if auto_widen else width1

    bend90 = (
        bend_factory(
            **waveguide_settings,
        )
        if callable(bend_factory)
        else bend_factory
    )

    taper = (
        taper_factory(
            length=taper_length,
            width1=input_port.width,
            width2=width2,
            waveguide=waveguide,
            **waveguide_settings,
        )
        if callable(taper_factory)
        else taper_factory
    )

    return route_manhattan(
        input_port=input_port,
        output_port=output_port,
        straight_factory=straight_factory,
        taper=taper,
        start_straight=start_straight,
        end_straight=end_straight,
        min_straight=min_straight,
        bend_factory=bend90,
        waveguide=waveguide,
        **waveguide_settings,
    )


def get_route_from_waypoints(
    waypoints: Coordinates,
    bend_factory: Callable = bend_euler,
    straight_factory: Callable = straight,
    taper_factory: Optional[Callable] = taper_function,
    waveguide: StrOrDict = "strip",
    route_filter=None,
    **waveguide_settings,
) -> Route:
    """Returns a route formed by the given waypoints with
    bends instead of corners and optionally tapers in straight sections.
    Tapering to wider straights reduces the optical loss.
    `get_route_from_waypoints` is a manual version of `get_route`
    Not that `get_route_from_steps` is a  more concise and convenient version of `get_route_from_waypoints` also available in gf.routing

    Args:
        waypoints: Coordinates that define the route
        bend_factory: function that returns bends
        straight_factory: function that returns straight waveguides
        taper_factory: function that returns tapers
        layer: for the route
        route_filter: FIXME, keep it here. Find a way to remove it.
        waveguide_settings

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component('waypoints_sample')

        w = gf.components.straight()
        left = c << w
        right = c << w
        right.move((100, 80))

        obstacle = gf.components.rectangle(size=(100, 10))
        obstacle1 = c << obstacle
        obstacle2 = c << obstacle
        obstacle1.ymin=40
        obstacle2.xmin=25


        p0x, p0y = left.ports['E0'].midpoint
        p1x, p1y = right.ports['E0'].midpoint
        o = 10 # vertical offset to overcome bottom obstacle
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
        c.show()
    """

    x = get_cross_section(waveguide, **waveguide_settings)
    waveguide_settings = x.info
    auto_widen = waveguide_settings.get("auto_widen", False)
    width1 = waveguide_settings.get("width")
    width2 = waveguide_settings.get("width_wide") if auto_widen else width1
    taper_length = waveguide_settings.get("taper_length")
    waypoints = np.array(waypoints)

    if auto_widen:
        taper = (
            taper_factory(
                length=taper_length,
                width1=width1,
                width2=width2,
                waveguide=waveguide,
                **waveguide_settings,
            )
            if callable(taper_factory)
            else taper_factory
        )
    else:
        taper = None

    return round_corners(
        points=waypoints,
        bend_factory=bend_factory,
        straight_factory=straight_factory,
        taper=taper,
        waveguide=waveguide,
        **waveguide_settings,
    )


if __name__ == "__main__":
    import gdsfactory as gf

    w = gf.components.mmi1x2()

    c = gf.Component()
    c << w
    # route = get_route(w.ports["E0"], w.ports["W0"], **gf.TECH.waveguide.nitride)
    # route = get_route(w.ports["E0"], w.ports["W0"], waveguide="metal_routing")
    route = get_route(w.ports["E0"], w.ports["W0"], layer=(2, 0))
    cc = c.add(route.references)
    cc.show()

"""Get route between two ports.

You can connect two ports with a manhattan route.

To make a route, you need to supply:

 - an input port
 - an output port
 - a bend or factory
 - a straight or factory
 - a taper or factory to taper to wider straights and reduce straight loss (optional)


To generate a straight route:

 1. Generate the backbone of the route.
 This is a list of manhattan coordinates that the route would pass through
 if it used only sharp bends (right angles)

 2. Replace the corners by bend references (with rotation and position computed from the manhattan backbone)

 3. Add tapers if needed and if space permits

 4. generate straight portions in between tapers or bends


 A route is a dict with 3 keys:

- references: list of references for tapers, bends and straight straights
- ports: a dict of port name to Port, usually two ports "input" and "output"
- length: a float with the length of the route

"""

from typing import Callable, Optional

import numpy as np
from numpy import ndarray

from pp.components import straight
from pp.components import taper as taper_function
from pp.components.bend_euler import bend_euler
from pp.components.electrical import corner, wire
from pp.config import TAPER_LENGTH, WG_EXPANDED_WIDTH
from pp.layers import LAYER
from pp.port import Port
from pp.routing.manhattan import round_corners, route_manhattan
from pp.snap import snap_to_grid
from pp.types import ComponentOrFactory, Coordinates, Layer, Number, Route


def get_route(
    input_port: Port,
    output_port: Port,
    bend_factory: ComponentOrFactory = bend_euler,
    straight_factory: ComponentOrFactory = straight,
    taper_factory: Optional[ComponentOrFactory] = taper_function,
    start_straight: Number = 0.01,
    end_straight: Number = 0.01,
    min_straight: Number = 0.01,
    bend_radius: Number = 10.0,
    auto_widen: bool = True,
) -> Route:
    """Returns a Route dict of references, ports and lengths.
    The references are straights, bends and tapers.

    Args:
        input_port: start port
        output_port: end port
        bend_factory: function that return bends
        straight_factory: function that returns straights
        start_straight: length of starting straight
        end_straight: Number: length of end straight
        min_straight: Number: min length of straight
        bend_radius: Number: min bend_radius
    """

    bend90 = (
        bend_factory(radius=bend_radius, width=input_port.width)
        if callable(bend_factory)
        else bend_factory
    )

    taper = (
        taper_factory(
            length=TAPER_LENGTH,
            width1=input_port.width,
            width2=WG_EXPANDED_WIDTH,
            layer=input_port.layer,
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
        auto_widen=auto_widen,
    )


def get_route_electrical(
    input_port: Port,
    output_port: Port,
    bend_factory: Callable = corner,
    straight_factory: Callable = wire,
    layer: Optional[Layer] = None,
    width: Optional[Number] = None,
    **kwargs
) -> Route:
    """Returns a Route dict of references, ports and lengths.
    The references are straights, bends and tapers.
    Uses electrical wire connectors by default

    Args:
        input_port: start port
        output_port: end port
        bend_factory: function that return bends
        straight_factory: function that returns straights
        layer: default wire layer
        width: defaul wire width
        start_straight: length of starting straight
        end_straight: Number: length of end straight
        min_straight: Number: min length of straight
        bend_radius: Number: min bend_radius
        route_factory: returns route
    """

    width = width or input_port.width
    layer = layer or input_port.layer

    def _bend_factory(width=width, radius=0):
        return bend_factory(width=width, radius=radius, layer=layer)

    def _straight_factory(length=10.0, width=width):
        return straight_factory(length=snap_to_grid(length), width=width, layer=layer)

    if "bend_radius" in kwargs:
        bend_radius = kwargs.pop("bend_radius")
    else:
        bend_radius = 0.001

    return get_route(
        input_port,
        output_port,
        bend_radius=bend_radius,
        bend_factory=_bend_factory,
        straight_factory=_straight_factory,
        taper_factory=None,
        **kwargs
    )


def get_route_from_waypoints(
    waypoints: Coordinates,
    bend_factory: Callable = bend_euler,
    straight_factory: Callable = straight,
    taper_factory: Optional[Callable] = taper_function,
    bend_radius: Number = 5.0,
    wg_width: Number = 0.5,
    layer: Layer = LAYER.WG,
    auto_widen: bool = True,
    **kwargs
) -> Route:
    """Returns a route formed by the given waypoints with
    bends instead of corners and optionally tapers in straight sections.
    Tapering to wider straights reduces the optical loss.

    Args:
        waypoints: Coordinates that define the route
        bend_factory: function that returns bends
        straight_factory: function that returns straight straights
        taper_factory: function that returns tapers
        bend_radius: of straights
        wg_width: for taper
        layer: for the route
    """
    waypoints = np.array(waypoints)
    bend90 = (
        bend_factory(radius=bend_radius, width=wg_width)
        if callable(bend_factory)
        else bend_factory
    )

    if auto_widen:
        taper = (
            taper_factory(
                length=TAPER_LENGTH,
                width1=wg_width,
                width2=WG_EXPANDED_WIDTH,
                layer=layer,
            )
            if callable(taper_factory)
            else taper_factory
        )
    else:
        taper = None

    return round_corners(
        points=waypoints,
        bend_factory=bend90,
        straight_factory=straight_factory,
        taper=taper,
        auto_widen=auto_widen,
    )


def get_route_from_waypoints_no_taper(*args, **kwargs) -> Route:
    """Returns route that does not taper to wider straights.

    Args:
        waypoints: Coordinates that define the route
        bend_factory: function that returns bends
        straight_factory: function that returns straight straights
        taper_factory: function that returns tapers
        bend_radius: of straights
        wg_width: for taper
        layer: for the route
    """
    return get_route_from_waypoints(*args, taper_factory=None, **kwargs)


def get_route_from_waypoints_electrical(
    waypoints: ndarray,
    bend_factory: Callable = corner,
    straight_factory: Callable = wire,
    taper_factory: Optional[Callable] = taper_function,
    wg_width: Number = 10.0,
    layer: Layer = LAYER.M3,
    **kwargs
) -> Route:
    """Returns route with electrical traces.

    Args:
        waypoints: Coordinates that define the route
        bend_factory: function that returns bends
        straight_factory: function that returns straight straights
        taper_factory: function that returns tapers
        wg_width: for taper
        layer: for the route

    """

    bend90 = bend_factory(width=wg_width, layer=layer)

    def _straight_factory(length=10.0, width=wg_width):
        return straight_factory(length=snap_to_grid(length), width=width)

    connector = round_corners(
        points=waypoints,
        straight_factory=_straight_factory,
        taper=None,
        bend_factory=bend90,
    )
    return connector


if __name__ == "__main__":
    import pp

    w = pp.components.mmi1x2()

    c = pp.Component()
    c << w
    route = get_route(w.ports["E0"], w.ports["W0"])
    cc = c.add(route["references"])
    cc.show()

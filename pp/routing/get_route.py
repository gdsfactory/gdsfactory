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

- references: list of references for tapers, bends and straight waveguides
- ports: a dict of port name to Port, usually two ports "input" and "output"
- length: a float with the length of the route

"""

from typing import Callable, Optional

import numpy as np
from numpy import ndarray

from pp.components.bend_euler import bend_euler
from pp.components.electrical.wire import corner, wire
from pp.components.straight import straight
from pp.components.taper import taper as taper_function
from pp.cross_section import get_waveguide_settings
from pp.port import Port
from pp.routing.manhattan import round_corners, route_manhattan
from pp.snap import snap_to_grid
from pp.types import ComponentOrFactory, Coordinates, Number, Route


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
        **waveguide_settings,
    )


def get_route_electrical(
    input_port: Port,
    output_port: Port,
    bend_factory: Callable = corner,
    straight_factory: Callable = wire,
    waveguide: str = "metal_routing",
    **waveguide_settings,
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
        route_factory: returns route
    """
    return get_route(
        input_port,
        output_port,
        bend_factory=bend_factory,
        straight_factory=straight_factory,
        taper_factory=None,
        waveguide=waveguide,
        **waveguide_settings,
    )


def get_route_from_waypoints(
    waypoints: Coordinates,
    bend_factory: Callable = bend_euler,
    straight_factory: Callable = straight,
    taper_factory: Optional[Callable] = taper_function,
    waveguide: str = "strip",
    route_filter=None,
    **waveguide_settings,
) -> Route:
    """Returns a route formed by the given waypoints with
    bends instead of corners and optionally tapers in straight sections.
    Tapering to wider straights reduces the optical loss.

    Args:
        waypoints: Coordinates that define the route
        bend_factory: function that returns bends
        straight_factory: function that returns straight waveguides
        taper_factory: function that returns tapers
        layer: for the route
        route_filter: FIXME, keep it here
        waveguide_settings
    """
    waveguide_settings = get_waveguide_settings(waveguide, **waveguide_settings)
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


def get_route_from_waypoints_no_taper(*args, **kwargs) -> Route:
    """Returns route that does not taper to wider straights.

    Args:
        waypoints: Coordinates that define the route
        bend_factory: function that returns bends
        straight_factory: function that returns straight waveguides
        taper_factory: function that returns tapers
        radius: of bends
        wg_width: for taper
        layer: for the route
    """
    return get_route_from_waypoints(*args, taper_factory=None, **kwargs)


def get_route_from_waypoints_electrical(
    waypoints: ndarray,
    bend_factory: Callable = corner,
    straight_factory: Callable = wire,
    taper_factory: Optional[Callable] = taper_function,
    waveguide: str = "metal_routing",
    **waveguide_settings,
) -> Route:
    """Returns route with electrical traces.

    Args:
        waypoints: Coordinates that define the route
        bend_factory: function that returns bends
        straight_factory: function that returns straight waveguides
        taper_factory: function that returns tapers
        wg_width: for taper
        layer: for the route

    """
    waveguide_settings = get_waveguide_settings(waveguide, **waveguide_settings)

    bend90 = bend_factory(**waveguide_settings)

    def _straight_factory(length=10.0, **waveguide_settings):
        return straight_factory(length=snap_to_grid(length), **waveguide_settings)

    connector = round_corners(
        points=waypoints,
        straight_factory=_straight_factory,
        taper=None,
        bend_factory=bend90,
        **waveguide_settings,
    )
    return connector


if __name__ == "__main__":
    import pp

    w = pp.components.mmi1x2()

    c = pp.Component()
    c << w
    # route = get_route(w.ports["E0"], w.ports["W0"], **pp.TECH.waveguide.nitride)
    route = get_route(w.ports["E0"], w.ports["W0"], waveguide="metal_routing")
    cc = c.add(route.references)
    cc.show()

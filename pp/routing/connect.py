from typing import Callable
import numpy as np
from numpy import ndarray

from pp.routing.manhattan import route_manhattan
from pp.routing.manhattan import generate_manhattan_waypoints
from pp.routing.manhattan import round_corners
from pp.components.bend_circular import bend_circular
from pp.components import waveguide
from pp.components import taper as taper_factory
from pp.components.electrical import wire, corner
from pp.component import ComponentReference

from pp.config import WG_EXPANDED_WIDTH, TAPER_LENGTH
from pp.layers import LAYER
from pp.port import Port


def get_waypoints_connect_strip(*args, **kwargs) -> ndarray:
    return connect_strip(*args, **kwargs, route_factory=generate_manhattan_waypoints)


def connect_strip(
    input_port: Port,
    output_port: Port,
    bend_factory: Callable = bend_circular,
    straight_factory: Callable = waveguide,
    taper_factory: Callable = taper_factory,
    start_straight: float = 0.01,
    end_straight: float = 0.01,
    min_straight: float = 0.01,
    bend_radius: float = 10.0,
    route_factory: Callable = route_manhattan,
) -> ComponentReference:
    """ return an optical route """

    bend90 = bend_factory(radius=bend_radius, width=input_port.width)

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

    connector = route_factory(
        input_port,
        output_port,
        bend90,
        straight_factory=straight_factory,
        taper=taper,
        start_straight=start_straight,
        end_straight=end_straight,
        min_straight=min_straight,
    )
    return connector


def connect_strip_way_points(
    way_points=[],
    bend_factory=bend_circular,
    straight_factory=waveguide,
    taper_factory=taper_factory,
    bend_radius=10.0,
    wg_width=0.5,
    layer=LAYER.WG,
    **kwargs
):
    """Returns a deep-etched route formed by the given way_points with
    bends instead of corners and optionally tapers in straight sections.

    taper_factory: can be either a taper Component or a factory
    """
    way_points = np.array(way_points)
    bend90 = bend_factory(radius=bend_radius, width=wg_width)

    taper = (
        taper_factory(
            length=TAPER_LENGTH, width1=wg_width, width2=WG_EXPANDED_WIDTH, layer=layer,
        )
        if callable(taper_factory)
        else taper_factory
    )

    connector = round_corners(way_points, bend90, straight_factory, taper)
    return connector


def connect_strip_way_points_no_taper(*args, **kwargs):
    return connect_strip_way_points(*args, taper_factory=None, **kwargs)


def connect_elec_waypoints(
    way_points=[],
    bend_factory=corner,
    straight_factory=wire,
    taper_factory=taper_factory,
    wg_width=10.0,
    layer=LAYER.M3,
    **kwargs
):
    """returns a route with electrical traces"""

    bend90 = bend_factory(width=wg_width, layer=layer)

    def _straight_factory(length=10.0, width=wg_width):
        return straight_factory(length=length, width=width, layer=layer)

    connector = round_corners(way_points, bend90, _straight_factory, taper=None)
    return connector


def connect_elec(
    input_port,
    output_port,
    straight_factory=wire,
    bend_factory=corner,
    layer=None,
    **kwargs
):
    width = input_port.width

    # If no layer is specified, the component factories should use the layer
    # given in port.type
    if layer is None:
        layer = input_port.layer

    def _bend_factory(width=width, radius=0):
        return bend_factory(width=width, radius=radius, layer=layer)

    def _straight_factory(length=10.0, width=width):
        return straight_factory(length=length, width=width, layer=layer)

    if "bend_radius" in kwargs:
        bend_radius = kwargs.pop("bend_radius")
    else:
        bend_radius = 10

    return connect_strip(
        input_port,
        output_port,
        bend_radius=bend_radius,
        bend_factory=_bend_factory,
        straight_factory=_straight_factory,
        taper_factory=None,
        **kwargs
    )


if __name__ == "__main__":
    import pp

    w = pp.c.mmi1x2()

    c = pp.Component()
    c << w
    connector_ref = connect_strip(w.ports["E0"], w.ports["W0"])
    cc = c.add(connector_ref)
    pp.show(cc)

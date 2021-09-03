"""Functions to create routes between components."""

from gdsfactory.routing import fanout, sort_ports, utils
from gdsfactory.routing.add_electrical_pads_shortest import add_electrical_pads_shortest
from gdsfactory.routing.add_electrical_pads_top import add_electrical_pads_top
from gdsfactory.routing.add_electrical_pads_top_dc import add_electrical_pads_top_dc
from gdsfactory.routing.add_fiber_array import add_fiber_array
from gdsfactory.routing.add_fiber_single import add_fiber_single
from gdsfactory.routing.fanout2x2 import fanout2x2
from gdsfactory.routing.get_bundle import get_bundle
from gdsfactory.routing.get_bundle_from_waypoints import get_bundle_from_waypoints
from gdsfactory.routing.get_bundle_path_length_match import get_bundle_path_length_match
from gdsfactory.routing.get_bundle_sbend import get_bundle_sbend
from gdsfactory.routing.get_route import (
    get_route,
    get_route_electrical,
    get_route_from_waypoints,
)
from gdsfactory.routing.get_route_from_steps import get_route_from_steps
from gdsfactory.routing.get_route_sbend import get_route_sbend
from gdsfactory.routing.get_routes_bend180 import get_routes_bend180
from gdsfactory.routing.get_routes_straight import get_routes_straight
from gdsfactory.routing.route_ports_to_side import route_ports_to_side
from gdsfactory.routing.route_south import route_south

__all__ = [
    "add_electrical_pads_shortest",
    "add_electrical_pads_top",
    "add_electrical_pads_top_dc",
    "add_fiber_array",
    "add_fiber_single",
    "get_bundle",
    "get_bundle_path_length_match",
    "get_bundle_from_waypoints",
    "get_route",
    "get_route_electrical",
    "get_routes_bend180",
    "get_routes_straight",
    "get_route_sbend",
    "get_bundle_sbend",
    "get_route_from_waypoints",
    "get_route_from_steps",
    "fanout2x2",
    "fanout",
    "route_ports_to_side",
    "route_south",
    "fanout",
    "sort_ports",
    "utils",
]


if __name__ == "__main__":
    print(__all__)

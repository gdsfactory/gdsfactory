"""Functions to create routes between components."""

import pp.routing.fanout as fanout
from pp.routing.add_electrical_pads_shortest import add_electrical_pads_shortest
from pp.routing.add_electrical_pads_top import add_electrical_pads_top
from pp.routing.add_fiber_array import add_fiber_array
from pp.routing.add_fiber_single import add_fiber_single
from pp.routing.fanout2x2 import fanout2x2
from pp.routing.get_bundle import get_bundle
from pp.routing.get_bundle_from_waypoints import get_bundle_from_waypoints
from pp.routing.get_bundle_path_length_match import get_bundle_path_length_match
from pp.routing.get_bundle_sbend import get_bundle_sbend
from pp.routing.get_route import get_route, get_route_from_waypoints
from pp.routing.get_route_sbend import get_route_sbend
from pp.routing.get_routes import get_routes
from pp.routing.get_routes_bend180 import get_routes_bend180
from pp.routing.route_ports_to_side import route_ports_to_side
from pp.routing.route_south import route_south

__all__ = [
    "add_electrical_pads_shortest",
    "add_electrical_pads_top",
    "add_fiber_array",
    "add_fiber_single",
    "get_bundle",
    "get_bundle_path_length_match",
    "get_bundle_from_waypoints",
    "get_route",
    "get_routes",
    "get_routes_bend180",
    "get_route_sbend",
    "get_bundle_sbend",
    "get_route_from_waypoints",
    "fanout2x2",
    "fanout",
    "route_ports_to_side",
    "route_south",
]


if __name__ == "__main__":
    print(__all__)

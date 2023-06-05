"""Functions to create routes between components."""

from __future__ import annotations

from gdsfactory.routing import sort_ports, utils
from gdsfactory.routing.add_electrical_pads_shortest import add_electrical_pads_shortest
from gdsfactory.routing.add_electrical_pads_top import add_electrical_pads_top
from gdsfactory.routing.add_electrical_pads_top_dc import add_electrical_pads_top_dc
from gdsfactory.routing.add_fiber_array import add_fiber_array
from gdsfactory.routing.add_fiber_single import add_fiber_single
from gdsfactory.routing.add_pads import add_pads_bot, add_pads_top
from gdsfactory.routing.all_angle import get_bundle_all_angle
from gdsfactory.routing.fanout import fanout_component, fanout_ports
from gdsfactory.routing.fanout2x2 import fanout2x2
from gdsfactory.routing.get_bundle import (
    get_bundle,
    get_bundle_electrical,
    get_bundle_electrical_multilayer,
)
from gdsfactory.routing.get_bundle_from_steps import (
    get_bundle_from_steps,
    get_bundle_from_steps_electrical,
    get_bundle_from_steps_electrical_multilayer,
)
from gdsfactory.routing.get_bundle_from_waypoints import (
    get_bundle_from_waypoints,
    get_bundle_from_waypoints_electrical,
    get_bundle_from_waypoints_electrical_multilayer,
)
from gdsfactory.routing.get_bundle_path_length_match import get_bundle_path_length_match
from gdsfactory.routing.get_bundle_sbend import get_bundle_sbend
from gdsfactory.routing.get_route import (
    get_route,
    get_route_electrical,
    get_route_electrical_m2,
    get_route_electrical_multilayer,
    get_route_from_waypoints,
    get_route_from_waypoints_electrical,
    get_route_from_waypoints_electrical_m2,
    get_route_from_waypoints_electrical_multilayer,
)
from gdsfactory.routing.get_route_astar import get_route_astar
from gdsfactory.routing.get_route_from_steps import (
    get_route_from_steps,
    get_route_from_steps_electrical,
    get_route_from_steps_electrical_multilayer,
)
from gdsfactory.routing.get_route_sbend import get_route_sbend
from gdsfactory.routing.get_routes_bend180 import get_routes_bend180
from gdsfactory.routing.get_routes_straight import get_routes_straight
from gdsfactory.routing.route_ports_to_side import route_ports_to_side
from gdsfactory.routing.route_quad import route_quad
from gdsfactory.routing.route_sharp import route_sharp
from gdsfactory.routing.route_south import route_south

__all__ = [
    "add_electrical_pads_shortest",
    "add_electrical_pads_top",
    "add_electrical_pads_top_dc",
    "add_pads_top",
    "add_pads_bot",
    "add_fiber_array",
    "add_fiber_single",
    "get_bundle",
    "get_bundle_all_angle",
    "get_bundle_from_steps",
    "get_bundle_from_steps_electrical",
    "get_bundle_from_steps_electrical_multilayer",
    "get_bundle_electrical",
    "get_bundle_electrical_multilayer",
    "get_bundle_path_length_match",
    "get_bundle_from_waypoints",
    "get_bundle_from_waypoints_electrical",
    "get_bundle_from_waypoints_electrical_multilayer",
    "get_route",
    "get_route_astar",
    "get_route_electrical",
    "get_route_electrical_m2",
    "get_route_electrical_multilayer",
    "get_routes_bend180",
    "get_routes_straight",
    "get_route_sbend",
    "get_bundle_sbend",
    "get_route_from_waypoints",
    "get_route_from_waypoints_electrical",
    "get_route_from_waypoints_electrical_m2",
    "get_route_from_waypoints_electrical_multilayer",
    "get_route_from_steps",
    "get_route_from_steps_electrical",
    "get_route_from_steps_electrical_multilayer",
    "fanout2x2",
    "fanout",
    "route_ports_to_side",
    "route_south",
    "route_quad",
    "route_sharp",
    "fanout_component",
    "fanout_ports",
    "sort_ports",
    "utils",
]


if __name__ == "__main__":
    print(__all__)

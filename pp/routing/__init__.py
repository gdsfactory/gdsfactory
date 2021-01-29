"""Get optical and electrical routes."""

from pp.routing.add_electrical_pads import add_electrical_pads
from pp.routing.add_electrical_pads_shortest import add_electrical_pads_shortest
from pp.routing.add_electrical_pads_top import add_electrical_pads_top
from pp.routing.add_fiber_array import add_fiber_array
from pp.routing.add_fiber_single import add_fiber_single
from pp.routing.get_bundle import (
    get_bundle,
    get_bundle_path_length_match,
    link_electrical_ports,
    link_optical_ports,
    link_optical_ports_no_grouping,
)
from pp.routing.get_bundle_from_waypoints import get_bundle_from_waypoints
from pp.routing.get_route import (
    get_route,
    get_route_electrical,
    get_route_from_waypoints,
    get_route_from_waypoints_electrical,
)
from pp.routing.manhattan import round_corners, route_manhattan
from pp.routing.repackage import package_optical2x2
from pp.routing.route_fiber_single import route_fiber_single
from pp.routing.route_ports_to_side import route_elec_ports_to_side, route_ports_to_side
from pp.routing.route_south import route_south

route_factory = dict(
    optical=get_route_from_waypoints, electrical=get_route_from_waypoints_electrical,
)

link_factory = dict(
    link_ports=get_bundle,
    link_ports_path_length_match=get_bundle_path_length_match,
    link_electrical_waypoints=get_route_from_waypoints_electrical,
    link_optical_waypoints=get_route_from_waypoints,
    link_optical_no_grouping=link_optical_ports_no_grouping,
)

__all__ = [
    "add_electrical_pads",
    "add_electrical_pads_shortest",
    "add_electrical_pads_top",
    "add_fiber_array",
    "add_fiber_array",
    "add_fiber_single",
    "get_bundle",
    "get_bundle_path_length_match",
    "get_bundle_from_waypoints",
    "get_route",
    "get_route_electrical",
    "get_route_from_waypoints",
    "link_electrical_ports",
    "link_optical_ports",
    "link_optical_ports_no_grouping",
    "link_factory",
    "package_optical2x2",
    "round_corners",
    "route_elec_ports_to_side",
    "route_fiber_single",
    "route_manhattan",
    "route_ports_to_side",
    "route_ports_to_side",
    "route_factory",
    "route_south",
]


if __name__ == "__main__":
    print(__all__)

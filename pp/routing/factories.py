"""Store route and link Factories in a dict.

FIXME. link_factory should have a route factory inside
"""

from pp.routing.get_bundle import (
    get_bundle,
    get_bundle_path_length_match,
    link_optical_ports_no_grouping,
)
from pp.routing.get_route import (
    get_route_from_waypoints,
    get_route_from_waypoints_electrical,
)

route_factory = dict(
    optical=get_route_from_waypoints,
    electrical=get_route_from_waypoints_electrical,
)

link_factory = dict(
    link_ports=get_bundle,
    link_ports_path_length_match=get_bundle_path_length_match,
    link_electrical_waypoints=get_route_from_waypoints_electrical,
    link_optical_waypoints=get_route_from_waypoints,
    link_optical_no_grouping=link_optical_ports_no_grouping,
)

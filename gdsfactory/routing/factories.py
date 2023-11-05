from __future__ import annotations

from gdsfactory.routing.all_angle import route_bundle_all_angle
from gdsfactory.routing.route_bundle import (
    route_bundle,
    route_bundle_electrical,
    route_bundle_same_axis_no_grouping,
)
from gdsfactory.routing.route_bundle_from_steps import (
    route_bundle_from_steps,
    route_bundle_from_steps_electrical,
)
from gdsfactory.routing.route_bundle_from_waypoints import route_bundle_from_waypoints
from gdsfactory.routing.route_bundle_path_length_match import (
    route_bundle_path_length_match,
)

routing_strategy = dict(
    route_bundle=route_bundle,
    route_bundle_electrical=route_bundle_electrical,
    route_bundle_path_length_match=route_bundle_path_length_match,
    route_bundle_same_axis_no_grouping=route_bundle_same_axis_no_grouping,
    route_bundle_from_waypoints=route_bundle_from_waypoints,
    route_bundle_from_steps=route_bundle_from_steps,
    route_bundle_from_steps_electrical=route_bundle_from_steps_electrical,
    route_bundle_all_angle=route_bundle_all_angle,
)

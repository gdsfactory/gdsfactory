from gdsfactory.routing.get_bundle import (
    get_bundle,
    get_bundle_electrical,
    get_bundle_same_axis_no_grouping,
)
from gdsfactory.routing.get_bundle_from_steps import (
    get_bundle_from_steps,
    get_bundle_from_steps_electrical,
)
from gdsfactory.routing.get_bundle_from_waypoints import get_bundle_from_waypoints
from gdsfactory.routing.get_bundle_path_length_match import get_bundle_path_length_match

routing_strategy = dict(
    get_bundle=get_bundle,
    get_bundle_electrical=get_bundle_electrical,
    get_bundle_path_length_match=get_bundle_path_length_match,
    get_bundle_same_axis_no_grouping=get_bundle_same_axis_no_grouping,
    get_bundle_from_waypoints=get_bundle_from_waypoints,
    get_bundle_from_steps=get_bundle_from_steps,
    get_bundle_from_steps_electrical=get_bundle_from_steps_electrical,
)

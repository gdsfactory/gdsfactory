from pp.routing.get_bundle import get_bundle, get_bundle_same_axis_no_grouping
from pp.routing.get_bundle_path_length_match import get_bundle_path_length_match
from pp.routing.get_route import get_route_from_waypoints

routing_strategy = dict(
    get_bundle=get_bundle,
    get_bundle_path_length_match=get_bundle_path_length_match,
    get_bundle_same_axis_no_grouping=get_bundle_same_axis_no_grouping,
    get_route_from_waypoints=get_route_from_waypoints,
)

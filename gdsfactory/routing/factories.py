from __future__ import annotations

from gdsfactory.routing.route_bundle import route_bundle, route_bundle_electrical
from gdsfactory.routing.route_bundle_all_angle import route_bundle_all_angle
from gdsfactory.routing.route_bundle_sbend import route_bundle_sbend
from gdsfactory.typings import RoutingStrategies

routing_strategies: RoutingStrategies = {
    "route_bundle": route_bundle,
    "route_bundle_electrical": route_bundle_electrical,
    "route_bundle_all_angle": route_bundle_all_angle,
    "route_bundle_sbend": route_bundle_sbend,
}

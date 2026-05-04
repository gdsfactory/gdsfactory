"""You can define a path as list of points.

To create a component you need to extrude the path with a cross-section.
"""

from functools import partial
from typing import Any, Protocol

import gdsfactory as gf
from doroutes.bundles import add_bundle_astar

route_single = partial(gf.routing.route_single, cross_section="strip")
route_bundle = partial(gf.routing.route_bundle, cross_section="strip")


class _RoutingStrategy(Protocol):
    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...


route_bundle_nitride = partial(route_bundle, cross_section="nitride")
route_bundle_rib = partial(route_bundle, cross_section="rib")
route_bundle_metal1 = partial(route_bundle, cross_section="metal1")
route_bundle_metal2 = partial(route_bundle, cross_section="metal2")
route_bundle_metal3 = partial(route_bundle, cross_section="metal3")

route_bundle_all_angle = partial(
    gf.routing.route_bundle_all_angle,
    cross_section="strip",
    separation=3,
    bend="bend_euler_all_angle",
    straight="straight_all_angle",
)

route_bundle_sbend = partial(
    gf.routing.route_bundle_sbend,
    cross_section="strip",
    bend_s=gf.components.bend_s,
)

route_bundle_sbend_nitride = partial(
    gf.routing.route_bundle_sbend,
    cross_section="nitride",
    bend_s=gf.components.bend_s,
)

route_bundle_sbend_metal3 = partial(
    gf.routing.route_bundle_sbend,
    cross_section="metal3",
    bend_s=gf.components.bend_s,
    port_name="e1",
)

route_astar = partial(
    add_bundle_astar,
    layers=["WG"],
    bend="bend_euler",
    straight="straight",
    grid_unit=500,
    spacing=3,
)

route_astar_metal3 = partial(
    add_bundle_astar,
    layers=["M3"],
    bend="wire_corner",
    straight="wire_straight",
    grid_unit=500,
    spacing=15,
)


routing_strategies: dict[str, _RoutingStrategy] = {
    "route_bundle": route_bundle,
    "route_bundle_nitride": route_bundle_nitride,
    "route_bundle_metal1": route_bundle_metal1,
    "route_bundle_metal2": route_bundle_metal2,
    "route_bundle_metal3": route_bundle_metal3,
    "route_bundle_sbend": route_bundle_sbend,
    "route_bundle_sbend_nitride": route_bundle_sbend_nitride,
    "route_bundle_sbend_metal3": route_bundle_sbend_metal3,
    "route_astar": route_astar,
    "route_astar_metal3": route_astar_metal3,
}

"""Functions to create routes between components."""

from __future__ import annotations

from gdsfactory.routing import sort_ports, utils
from gdsfactory.routing.add_electrical_pads_shortest import add_electrical_pads_shortest
from gdsfactory.routing.add_electrical_pads_top import add_electrical_pads_top
from gdsfactory.routing.add_electrical_pads_top_dc import add_electrical_pads_top_dc
from gdsfactory.routing.add_fiber_array import add_fiber_array
from gdsfactory.routing.add_fiber_single import add_fiber_single
from gdsfactory.routing.add_pads import add_pads_bot, add_pads_top
from gdsfactory.routing.fanout2x2 import fanout2x2
from gdsfactory.routing.route_bundle import (
    route_bundle,
    route_bundle_electrical,
)
from gdsfactory.routing.route_bundle_all_angle import route_bundle_all_angle
from gdsfactory.routing.route_bundle_sbend import route_bundle_sbend
from gdsfactory.routing.route_ports_to_side import route_ports_to_side
from gdsfactory.routing.route_quad import route_quad
from gdsfactory.routing.route_sharp import route_sharp
from gdsfactory.routing.route_single import (
    route_single,
    route_single_electrical,
)
from gdsfactory.routing.route_single_from_steps import (
    route_single_from_steps,
    route_single_from_steps_electrical,
)
from gdsfactory.routing.route_single_sbend import route_single_sbend
from gdsfactory.routing.route_south import route_south

__all__ = [
    "add_electrical_pads_shortest",
    "add_electrical_pads_top",
    "add_electrical_pads_top_dc",
    "add_pads_top",
    "add_pads_bot",
    "add_fiber_array",
    "add_fiber_single",
    "route_bundle",
    "route_bundle_all_angle",
    "route_bundle_electrical",
    "route_single",
    "route_single_electrical",
    "route_bundle_sbend",
    "route_single_from_steps",
    "route_single_from_steps_electrical",
    "fanout2x2",
    "route_ports_to_side",
    "route_south",
    "route_quad",
    "route_sharp",
    "sort_ports",
    "utils",
    "route_single_electrical",
    "route_bundle",
    "route_single_from_steps",
    "route_single_sbend",
]


if __name__ == "__main__":
    print(__all__)

"""Containers are components that contain other components."""

from gdsfactory.add_padding import add_padding_to_size, add_padding_to_size_container
from gdsfactory.components import (
    add_fiber_array_optical_south_electrical_north,
    add_termination,
    add_trenches,
    add_trenches90,
    array,
    cutback_bend,
    cutback_bend90,
    cutback_bend90circular,
    cutback_bend180,
    cutback_bend180circular,
    cutback_component,
    cutback_component_mirror,
    cutback_loss,
    cutback_loss_bend90,
    cutback_loss_bend180,
    cutback_loss_mmi1x2,
    cutback_loss_spirals,
    cutback_splitter,
    extend_ports,
    pack_doe,
    pack_doe_grid,
    staircase,
)
from gdsfactory.components.pcms.cutback_2x2 import cutback_2x2
from gdsfactory.functions import (
    extract,
    move_port_to_zero,
    rotate90,
    rotate180,
    rotate270,
    trim,
)
from gdsfactory.routing.add_electrical_pads_shortest import add_electrical_pads_shortest
from gdsfactory.routing.add_electrical_pads_top import add_electrical_pads_top
from gdsfactory.routing.add_electrical_pads_top_dc import add_electrical_pads_top_dc
from gdsfactory.routing.add_fiber_array import add_fiber_array
from gdsfactory.routing.add_fiber_single import add_fiber_single
from gdsfactory.routing.add_pads import add_pads_bot, add_pads_top
from gdsfactory.routing.fanout2x2 import fanout2x2
from gdsfactory.routing.route_ports_to_side import route_ports_to_side
from gdsfactory.routing.route_south import route_south

__all__ = [
    "add_electrical_pads_shortest",
    "add_electrical_pads_top",
    "add_electrical_pads_top_dc",
    "add_fiber_array",
    "add_fiber_array_optical_south_electrical_north",
    "add_fiber_single",
    "add_padding_to_size",
    "add_padding_to_size_container",
    "add_pads_bot",
    "add_pads_top",
    "add_termination",
    "add_trenches",
    "add_trenches90",
    "array",
    "cutback_2x2",
    "cutback_bend",
    "cutback_bend90",
    "cutback_bend90circular",
    "cutback_bend180",
    "cutback_bend180circular",
    "cutback_component",
    "cutback_component_mirror",
    "cutback_loss",
    "cutback_loss_bend90",
    "cutback_loss_bend180",
    "cutback_loss_mmi1x2",
    "cutback_loss_spirals",
    "cutback_splitter",
    "extend_ports",
    "extract",
    "fanout2x2",
    "move_port_to_zero",
    "pack_doe",
    "pack_doe_grid",
    "rotate90",
    "rotate180",
    "rotate270",
    "route_ports_to_side",
    "route_south",
    "staircase",
    "trim",
]

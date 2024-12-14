"""Containers are components that contain other components."""

from gdsfactory.add_padding import add_padding_to_size, add_padding_to_size_container
from gdsfactory.components.add_fiber_array_optical_south_electrical_north import (
    add_fiber_array_optical_south_electrical_north,
)
from gdsfactory.components.add_termination import add_termination
from gdsfactory.components.add_trenches import add_trenches, add_trenches90
from gdsfactory.components.array_component import array
from gdsfactory.components.cutback_2x2 import cutback_2x2
from gdsfactory.components.cutback_bend import (
    cutback_bend,
    cutback_bend90,
    cutback_bend90circular,
    cutback_bend180,
    cutback_bend180circular,
    staircase,
)
from gdsfactory.components.cutback_component import (
    cutback_component,
    cutback_component_mirror,
)
from gdsfactory.components.cutback_loss import (
    cutback_loss,
    cutback_loss_bend90,
    cutback_loss_bend180,
    cutback_loss_mmi1x2,
    cutback_loss_spirals,
)
from gdsfactory.components.cutback_splitter import cutback_splitter
from gdsfactory.components.extension import extend_ports
from gdsfactory.components.pack_doe import pack_doe, pack_doe_grid
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

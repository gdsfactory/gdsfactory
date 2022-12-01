from __future__ import annotations

import sys

from gdsfactory.add_padding import add_padding_container
from gdsfactory.add_termination import add_termination
from gdsfactory.components.add_grating_couplers import (
    add_grating_couplers,
    add_grating_couplers_with_loopback_fiber_single,
)
from gdsfactory.components.array_component import array
from gdsfactory.components.bend_port import bend_port
from gdsfactory.components.cavity import cavity
from gdsfactory.components.extension import extend_ports
from gdsfactory.components.pack_doe import pack_doe, pack_doe_grid
from gdsfactory.components.ring_single_dut import ring_single_dut
from gdsfactory.functions import rotate
from gdsfactory.get_factories import get_cells
from gdsfactory.routing.add_electrical_pads_shortest import add_electrical_pads_shortest
from gdsfactory.routing.add_electrical_pads_top import add_electrical_pads_top
from gdsfactory.routing.add_fiber_array import add_fiber_array
from gdsfactory.routing.add_fiber_single import add_fiber_single
from gdsfactory.routing.fanout2x2 import fanout2x2

__all__ = [
    "add_electrical_pads_shortest",
    "add_electrical_pads_top",
    "add_fiber_array",
    "add_fiber_single",
    "add_grating_couplers",
    "add_grating_couplers_with_loopback_fiber_single",
    "add_padding_container",
    "add_termination",
    "array",
    "bend_port",
    "cavity",
    "extend_ports",
    "fanout2x2",
    "pack_doe",
    "pack_doe_grid",
    "ring_single_dut",
    "rotate",
]


containers = get_cells(sys.modules[__name__])

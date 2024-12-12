"""Containers are Pcells that contain another Pcell."""

from __future__ import annotations

from gdsfactory.add_padding import add_padding_container
from gdsfactory.components import add_termination
from gdsfactory.routing import add_pads_bot, add_pads_top
from gdsfactory.routing.add_electrical_pads_shortest import add_electrical_pads_shortest
from gdsfactory.routing.add_electrical_pads_top import add_electrical_pads_top
from gdsfactory.routing.add_fiber_array import add_fiber_array
from gdsfactory.routing.fanout2x2 import fanout2x2

__all__ = [
    "add_electrical_pads_shortest",
    "add_electrical_pads_top",
    "add_fiber_array",
    "add_padding_container",
    "add_pads_bot",
    "add_pads_top",
    "add_termination",
    "fanout2x2",
]

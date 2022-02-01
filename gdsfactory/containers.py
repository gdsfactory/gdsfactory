from gdsfactory.add_grating_couplers import (
    add_grating_couplers,
    add_grating_couplers_with_loopback_fiber_single,
)
from gdsfactory.add_padding import add_padding_container
from gdsfactory.add_termination import add_termination
from gdsfactory.components.bend_port import bend_port
from gdsfactory.components.cavity import cavity
from gdsfactory.components.extension import extend_ports
from gdsfactory.components.ring_single_dut import ring_single_dut
from gdsfactory.functions import rotate
from gdsfactory.routing.add_electrical_pads_shortest import add_electrical_pads_shortest
from gdsfactory.routing.add_electrical_pads_top import add_electrical_pads_top
from gdsfactory.routing.add_fiber_array import add_fiber_array
from gdsfactory.routing.add_fiber_single import add_fiber_single
from gdsfactory.routing.fanout2x2 import fanout2x2

container_factory = dict(
    add_electrical_pads_shortest=add_electrical_pads_shortest,
    add_electrical_pads_top=add_electrical_pads_top,
    add_fiber_array=add_fiber_array,
    add_fiber_single=add_fiber_single,
    add_grating_couplers=add_grating_couplers,
    add_grating_couplers_with_loopback_fiber_single=add_grating_couplers_with_loopback_fiber_single,
    # add_grating_couplers_with_loopback_fiber_array=add_grating_couplers_with_loopback_fiber_array,
    add_padding_container=add_padding_container,
    add_termination=add_termination,
    cavity=cavity,
    extend_ports=extend_ports,
    fanout2x2=fanout2x2,
    ring_single_dut=ring_single_dut,
    bend_port=bend_port,
    rotate=rotate,
)


__all__ = list(container_factory.keys())

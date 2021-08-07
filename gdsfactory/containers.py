from gdsfactory.add_grating_couplers import add_grating_couplers
from gdsfactory.add_padding import add_padding_container
from gdsfactory.add_pins import add_pins_container
from gdsfactory.add_tapers import add_tapers
from gdsfactory.add_termination import add_gratings_and_loopback, add_termination
from gdsfactory.components.cavity import cavity
from gdsfactory.components.extension import extend_ports
from gdsfactory.components.ring_single_dut import ring_single_dut
from gdsfactory.rotate import rotate
from gdsfactory.routing.add_electrical_pads_shortest import add_electrical_pads_shortest
from gdsfactory.routing.add_electrical_pads_top import add_electrical_pads_top
from gdsfactory.routing.add_fiber_array import add_fiber_array
from gdsfactory.routing.add_fiber_single import add_fiber_single
from gdsfactory.routing.fanout2x2 import fanout2x2
from gdsfactory.tech import Library

COMPONENT_FACTORY = Library(name="generic_containers")
COMPONENT_FACTORY.register(
    [
        add_electrical_pads_shortest,
        add_electrical_pads_top,
        add_fiber_array,
        add_fiber_single,
        add_grating_couplers,
        add_gratings_and_loopback,
        add_padding_container,
        add_pins_container,
        add_tapers,
        add_termination,
        cavity,
        extend_ports,
        fanout2x2,
        ring_single_dut,
        rotate,
    ]
)


__all__ = list(COMPONENT_FACTORY.factory.keys())

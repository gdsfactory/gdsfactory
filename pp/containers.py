from pp.add_grating_couplers import add_grating_couplers
from pp.add_padding import add_padding_container
from pp.add_pins import add_pins_container
from pp.add_tapers import add_tapers
from pp.add_termination import add_gratings_and_loop_back, add_termination
from pp.components.cavity import cavity
from pp.components.extension import extend_ports
from pp.components.ring_single_dut import ring_single_dut
from pp.rotate import rotate
from pp.routing.add_electrical_pads_shortest import add_electrical_pads_shortest
from pp.routing.add_electrical_pads_top import add_electrical_pads_top
from pp.routing.add_fiber_array import add_fiber_array
from pp.routing.add_fiber_single import add_fiber_single
from pp.routing.fanout2x2 import fanout2x2
from pp.tech import Library

COMPONENT_FACTORY = Library(name="generic_containers")
COMPONENT_FACTORY.register(
    [
        add_electrical_pads_shortest,
        add_electrical_pads_top,
        add_fiber_array,
        add_fiber_single,
        add_grating_couplers,
        add_gratings_and_loop_back,
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

import pytest

from pp.add_grating_couplers import add_grating_couplers
from pp.add_padding import add_padding
from pp.add_pins import add_pins
from pp.add_tapers import add_tapers
from pp.add_termination import add_gratings_and_loop_back, add_termination

# from pp.components.waveguide import waveguide
# from pp.components.waveguide_heater import waveguide_heater
from pp.components import mzi2x2
from pp.components.extension import extend_ports
from pp.components.spiral_inner_io import spiral_inner_io
from pp.rotate import rotate
from pp.routing import (
    add_electrical_pads,
    add_electrical_pads_shortest,
    add_electrical_pads_top,
    add_fiber_array,
    add_fiber_single,
    package_optical2x2,
)

container_factory = dict(
    add_electrical_pads=add_electrical_pads,
    add_electrical_pads_shortest=add_electrical_pads_shortest,
    add_electrical_pads_top=add_electrical_pads_top,
    add_fiber_array=add_fiber_array,
    add_fiber_single=add_fiber_single,
    add_grating_couplers=add_grating_couplers,
    add_padding=add_padding,
    add_tapers=add_tapers,
    add_termination=add_termination,
    add_pins=add_pins,
    extend_ports=extend_ports,
    package_optical2x2=package_optical2x2,
    rotate=rotate,
)


def test_add_gratings_and_loop_back(data_regression):
    c = add_gratings_and_loop_back(component=spiral_inner_io())
    data_regression.check(c.get_settings())


@pytest.mark.parametrize("container_type", container_factory.keys())
def test_properties_containers(container_type, data_regression):
    component = mzi2x2(with_elec_connections=True)
    function = container_factory[container_type]
    c = function(component=component)
    data_regression.check(c.get_settings())

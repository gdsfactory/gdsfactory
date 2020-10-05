import pytest

from pp.components.extension import extend_ports
from pp.add_padding import add_padding
from pp.add_tapers import add_tapers
from pp.rotate import rotate
from pp.add_grating_couplers import add_grating_couplers
from pp.add_termination import add_termination, add_gratings_and_loop_back
from pp.components.spiral_inner_io import spiral_inner_io

# from pp.components.waveguide import waveguide
# from pp.components.waveguide_heater import waveguide_heater
from pp.components import mzi2x2

from pp.routing import add_fiber_array
from pp.routing import add_fiber_single
from pp.routing import add_electrical_pads
from pp.routing import add_electrical_pads_top
from pp.routing import add_electrical_pads_shortest
from pp.routing import package_optical2x2


_containers = [
    extend_ports,
    add_padding,
    add_tapers,
    rotate,
    add_termination,
    add_fiber_single,
    add_fiber_array,
    add_electrical_pads,
    add_electrical_pads_top,
    add_electrical_pads_shortest,
    add_grating_couplers,
    package_optical2x2,
]


def test_add_gratings_and_loop_back(data_regression):
    c = add_gratings_and_loop_back(component=spiral_inner_io())
    data_regression.check(c.get_settings())


@pytest.mark.parametrize("function", _containers)
def test_properties_containers(function, data_regression):
    component = mzi2x2(with_elec_connections=True)
    c = function(component=component)
    data_regression.check(c.get_settings())

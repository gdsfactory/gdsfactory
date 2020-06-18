import pytest

from pp.components.extension import extend_ports
from pp.add_padding import add_padding
from pp.routing.connect_component import add_io_optical
from pp.add_tapers import add_tapers, add_tapers2
from pp.rotate import rotate
from pp.add_termination import add_termination, add_gratings_and_loop_back
from pp.components.spiral_inner_io import spiral_inner_io
from pp.components.waveguide import waveguide


_containers = [
    extend_ports,
    add_padding,
    add_io_optical,
    add_tapers2,
    add_tapers,
    rotate,
    add_termination,
]


def test_add_gratings_and_loop_back(data_regression):
    c = add_gratings_and_loop_back(component=spiral_inner_io())
    data_regression.check(c.get_settings())


@pytest.mark.parametrize("function", _containers)
def test_properties_containers(function, data_regression):
    c = function(component=waveguide())
    data_regression.check(c.get_settings())

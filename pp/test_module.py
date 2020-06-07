# this code has been automatically generated from pp/components/__init__.py

import pp
from pp.components.extension import extend_ports
from pp.add_padding import add_padding
from pp.routing.connect_component import add_io_optical
from pp.add_tapers import add_tapers, add_tapers2
from pp.rotate import rotate
from pp.add_termination import add_termination, add_gratings_and_loop_back
from pp.components.spiral_inner_io import spiral_inner_io


def test_add_gratings_and_loop_back(data_regression):
    c = add_gratings_and_loop_back(component=spiral_inner_io())
    data_regression.check(c.get_settings())


def test_extend_ports(data_regression):
    c = extend_ports(component=pp.c.waveguide())
    data_regression.check(c.get_settings())


def test_add_padding(data_regression):
    c = add_padding(component=pp.c.waveguide())
    data_regression.check(c.get_settings())


def test_add_io_optical(data_regression):
    c = add_io_optical(component=pp.c.waveguide())
    data_regression.check(c.get_settings())


def test_add_tapers2(data_regression):
    c = add_tapers2(component=pp.c.waveguide())
    data_regression.check(c.get_settings())


def test_add_tapers(data_regression):
    c = add_tapers(component=pp.c.waveguide())
    data_regression.check(c.get_settings())


def test_rotate(data_regression):
    c = rotate(component=pp.c.waveguide())
    data_regression.check(c.get_settings())


def test_add_termination(data_regression):
    c = add_termination(component=pp.c.waveguide())
    data_regression.check(c.get_settings())

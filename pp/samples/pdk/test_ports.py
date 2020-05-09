# this code has been automatically generated from pp/samples/pdk/write_tests.py
import pp.samples.pdk as pdk


def test_mzi(num_regression):
    c = pdk.mzi()
    num_regression.check(c.get_ports_array())


def test_waveguide(num_regression):
    c = pdk.waveguide()
    num_regression.check(c.get_ports_array())


def test_bend_circular(num_regression):
    c = pdk.bend_circular()
    num_regression.check(c.get_ports_array())


def test_y_splitter(num_regression):
    c = pdk.y_splitter()
    num_regression.check(c.get_ports_array())

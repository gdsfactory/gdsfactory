# this code has been automatically generated from pp/samples/pdk/write_tests.py
import pp.samples.pdk as pdk


def test_waveguide(data_regression):
    c = pdk.waveguide()
    data_regression.check(c.get_settings())


def test_bend_circular(data_regression):
    c = pdk.bend_circular()
    data_regression.check(c.get_settings())


def test_y_splitter(data_regression):
    c = pdk.y_splitter()
    data_regression.check(c.get_settings())


def test_mzi(data_regression):
    c = pdk.mzi()
    data_regression.check(c.get_settings())

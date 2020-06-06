# this code has been automatically generated from pp/components/__init__.py
import pp


def test_cavity(data_regression):
    c = pp.c.cavity(component=pp.c.waveguide())
    data_regression.check(c.get_settings())

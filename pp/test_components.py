import pytest

from pp.components import component_factory, component_names, component_names_test_ports
from pp.testing import difftest


@pytest.mark.parametrize("component_type", component_names)
def test_gds(component_type, data_regression):
    """Avoid regressions in GDS geometry shapes and layers."""
    c = component_factory[component_type]()
    difftest(c)


@pytest.mark.parametrize("component_type", component_names)
def test_settings(component_type, data_regression):
    """Avoid regressions when exporting settings."""
    c = component_factory[component_type]()
    data_regression.check(c.get_settings())


@pytest.mark.parametrize("component_type", component_names_test_ports)
def test_ports(component_type, num_regression):
    """Avoid regressions in port names and locations."""
    c = component_factory[component_type]()
    if c.ports:
        num_regression.check(c.get_ports_array())

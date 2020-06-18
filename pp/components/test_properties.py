import pytest
from pp.components import (
    component_type2factory,
    _components,
    _containers,
    waveguide,
)


@pytest.mark.parametrize("component_type", _components)
def test_components_ports(component_type, num_regression):
    c = component_type2factory[component_type]()
    if c.ports:
        num_regression.check(c.get_ports_array())


@pytest.mark.parametrize("component_type", _components)
def test_properties_components(component_type, data_regression):
    c = component_type2factory[component_type]()
    data_regression.check(c.get_settings())


@pytest.mark.parametrize("component_type", _containers)
def test_properties_containers(component_type, data_regression):
    c = component_type2factory[component_type](component=waveguide())
    data_regression.check(c.get_settings())

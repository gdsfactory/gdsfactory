import pytest
from pp.samples.pdk import _components, component_type2factory


@pytest.mark.parametrize("component_type", _components)
def test_properties(component_type, data_regression):
    c = component_type2factory[component_type]()
    data_regression.check(c.get_settings())


@pytest.mark.parametrize("component_type", _components)
def test_ports(component_type, num_regression):
    c = component_type2factory[component_type]()
    if c.ports:
        num_regression.check(c.get_ports_array())


if __name__ == "__main__":
    test_properties()

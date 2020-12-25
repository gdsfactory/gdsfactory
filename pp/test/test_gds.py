import pytest

import pp
from pp.components import _components, component_factory
from pp.test_containers import container_factory
from pp.testing import difftest

components = _components - {"test_via", "test_resistance", "coupler"}
containers = set(container_factory.keys())


@pytest.mark.parametrize("component_type", components)
def test_gds_components(component_type):
    pp.clear_cache()
    component = component_factory[component_type]()
    difftest(component)


@pytest.mark.parametrize("component_type", containers)
def test_gds_containers(component_type):
    pp.clear_cache()
    component_inside_container = pp.c.mzi2x2(with_elec_connections=True)
    component = container_factory[component_type](component=component_inside_container)
    difftest(component)

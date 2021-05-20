"""FIXME

Test class for all the methods.
"""


import pytest

from pp.component import Component
from pp.difftest import TestComponent
from pp.samples.pdk.fab_c import COMPONENT_FACTORY
from pp.samples.pdk.fab_c import TECH_FABC as TECH

component_factory = COMPONENT_FACTORY.factory
component_names = component_factory.keys()


@pytest.fixture(params=component_names, scope="function")
def component(request) -> Component:
    return component_factory[request.param](cache=False)


class TestFabC(TestComponent):
    name: str = TECH.name


if __name__ == "__main__":
    t = TestComponent()
    t.test_gds(component=COMPONENT_FACTORY.get_component("mzi_nitride_cband"))

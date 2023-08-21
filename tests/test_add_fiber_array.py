from __future__ import annotations

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.difftest import difftest


def type0() -> Component:
    component = gf.components.coupler(gap=0.2, length=5.99)
    return gf.routing.add_fiber_array(component=component, optical_routing_type=0)


def type1() -> Component:
    component = gf.components.coupler(gap=0.2, length=5.0)
    return gf.routing.add_fiber_array(component=component, optical_routing_type=1)


def type2() -> Component:
    c = gf.components.coupler(gap=0.244, length=5.67)
    return gf.routing.add_fiber_array(component=c, optical_routing_type=2)


def tapers() -> Component:
    c = gf.components.straight(width=2, length=20)
    cc = gf.add_tapers(component=c)
    return gf.routing.add_fiber_array(component=cc, optical_routing_type=0)


components = [type0, type1, type2, tapers]


@pytest.fixture(params=components, scope="function")
def component(request) -> Component:
    return request.param()


def test_gds(component: Component) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    difftest(component)


def test_settings(component: Component, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    data_regression.check(component.to_dict())


if __name__ == "__main__":
    # c = type1()
    c = type2()
    # c = tapers()
    c.show(show_ports=True)

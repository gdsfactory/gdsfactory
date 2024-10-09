from __future__ import annotations

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.difftest import difftest


@gf.cell
def type1() -> Component:
    component = gf.components.coupler(gap=0.2, length=5.0)
    return gf.routing.add_fiber_array(component=component, optical_routing_type=1)


@gf.cell
def type2() -> Component:
    c = gf.components.coupler(gap=0.244, length=5.67)
    return gf.routing.add_fiber_array(component=c, optical_routing_type=2)


@gf.cell(check_ports=False)  # TODO: fix port auto naming
def fiber_single() -> Component:
    c = gf.components.coupler(gap=0.244, length=5.67)
    return gf.routing.add_fiber_single(component=c)


components = [type1, type2, fiber_single]


@pytest.fixture(params=components, scope="function")
def component(request) -> Component:
    return request.param()


def test_gds(component: Component) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    difftest(component)


def test_settings(component: Component, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    settings = component.to_dict()
    data_regression.check(settings)


if __name__ == "__main__":
    # c = type1()
    # c = type2()
    # c = tapers()
    c = fiber_single()
    c.pprint_ports()
    c.show()

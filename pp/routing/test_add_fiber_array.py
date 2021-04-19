import pytest
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture

import pp
from pp.component import Component
from pp.difftest import difftest


def test_type0() -> Component:
    component = pp.components.coupler(gap=0.244, length=5.67)
    cc = pp.routing.add_fiber_array(component=component, optical_routing_type=0)
    return cc


def test_type1() -> Component:
    component = pp.components.coupler(gap=0.2, length=5.0)
    cc = pp.routing.add_fiber_array(component=component, optical_routing_type=1)
    return cc


def test_type2() -> Component:
    c = pp.components.coupler(gap=0.244, length=5.67)
    c.polarization = "tm"
    cc = pp.routing.add_fiber_array(component=c, optical_routing_type=2)
    return cc


def test_tapers():
    c = pp.components.straight(width=2)
    cc = pp.routing.add_fiber_array(component=c, optical_routing_type=2)
    return cc


components = [test_type0, test_type1, test_type2, test_tapers]


@pytest.fixture(params=components, scope="function")
def component(request) -> Component:
    return request.param()


def test_gds(component: Component) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    difftest(component)


def test_settings(component: Component, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    data_regression.check(component.get_settings())


def test_ports(component: Component, num_regression: NumericRegressionFixture) -> None:
    """Avoid regressions in port names and locations."""
    if component.ports:
        num_regression.check(component.get_ports_array())


if __name__ == "__main__":
    c = test_type1()
    c.show()

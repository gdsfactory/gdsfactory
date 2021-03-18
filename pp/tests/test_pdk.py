import pytest
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture

from pp.component import Component
from pp.difftest import difftest
from pp.pdk import PDK_NITRIDE_C

pdk = PDK_NITRIDE_C

component_factory = pdk.get_factory_functions()


@pytest.fixture(params=component_factory.keys(), scope="function")
def component(request) -> Component:
    function = component_factory[request.param]
    return function()


def test_pdk_gds(component: Component) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    difftest(component)


def test_pdk_settings(
    component: Component, data_regression: DataRegressionFixture
) -> None:
    """Avoid regressions when exporting settings."""
    data_regression.check(component.get_settings())


def test_pdk_ports(
    component: Component, num_regression: NumericRegressionFixture
) -> None:
    """Avoid regressions in port names and locations."""
    if component.ports:
        num_regression.check(component.get_ports_array())

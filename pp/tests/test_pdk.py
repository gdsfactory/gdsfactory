import pytest
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture

from pp.component import Component
from pp.difftest import difftest
from pp.pdk import PDK_NITRIDE_C

pdk = PDK_NITRIDE_C

# All functions that do not start with (get, _, add) are a component_factory
component_names = [
    function_name
    for function_name in dir(pdk)
    if not function_name.startswith("get_")
    and not function_name.startswith("_")
    and not function_name.startswith("add_")
    and not function_name.startswith("tech")
]


@pytest.fixture(params=component_names, scope="function")
def component(request) -> Component:
    function = getattr(pdk, request.param)
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

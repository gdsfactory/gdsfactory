import pytest
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture

from gdsfactory.component import Component
from gdsfactory.components import factory
from gdsfactory.difftest import difftest

skip_test = {
    "version_stamp",
    "extend_ports_list",
    "extend_port",
    "component_sequence",
    "mzi_arm",
}

components_to_test = set(factory.keys()) - skip_test


@pytest.fixture(params=components_to_test, scope="function")
def component(request) -> Component:
    return factory[request.param]()


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


def test_assert_ports_on_grid(component: Component):
    """Ensure ports are on grid."""
    component.assert_ports_on_grid()


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.coupler(length=1.0 + 1e-4)
    # c = gf.components.coupler()
    # c.assert_ports_on_grid()
    print(c.ports)

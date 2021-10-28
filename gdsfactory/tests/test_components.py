import pathlib

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

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
dirpath = pathlib.Path(__file__).absolute().with_suffix(".gds")


@pytest.fixture(params=components_to_test, scope="function")
def component_name(request) -> str:
    return request.param


def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    component = factory[component_name]()
    difftest(component, test_name=component_name, dirpath=dirpath)


def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    component = factory[component_name]()
    data_regression.check(component.to_dict())


def test_assert_ports_on_grid(component_name: str):
    """Ensure ports are on grid."""
    component = factory[component_name]()
    component.assert_ports_on_grid()


if __name__ == "__main__":
    # c = gf.components.coupler(length=1.0 + 1e-4)
    # c = gf.components.coupler()
    # c.assert_ports_on_grid()
    # print(c.ports)
    print(dirpath)

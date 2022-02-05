"""Test all the components in fab_c.

In gdsfactory we use gdslib as a place to store the GDS files.

For your PDK i recommend that you store the store the reference GDS files on
the same repo as you store the code. See code below

```
import pathlib

dirpath = pathlib.Path(__file__).absolute().with_suffix(".gds")

def test_gds(component_name: str) -> None:
    component = factory[component_name]()
    test_name = f"fabc_{component_name}"
    difftest(component, test_name=test_name, dirpath=dirpath)

```

"""


import pytest
from pytest_regressions.data_regression import DataRegressionFixture

from gdsfactory.difftest import difftest
from gdsfactory.samples.pdk.fab_c import factory

component_names = list(factory.keys())


@pytest.fixture(params=component_names, scope="function")
def component_name(request) -> str:
    return request.param


def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS names, shapes and layers.
    Runs XOR and computes the area."""
    component = factory[component_name]()
    test_name = f"fabc_{component_name}"
    difftest(component, test_name=test_name)


def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions in component settings and ports."""
    component = factory[component_name]()
    data_regression.check(component.to_dict())


def test_assert_ports_on_grid(component_name: str):
    """Ensures all ports are on grid to avoid 1nm gaps"""
    component = factory[component_name]()
    component.assert_ports_on_grid()


if __name__ == "__main__":
    print(component_names)
    c = factory[component_names[0]]()
    difftest(c, test_name=f"fabc_{component_names[0]}")

"""Test all the components in fab_c.

In gdsfactory we use gdslib as a place to store the GDS files.

For your PDK i recommend that you store the store the reference GDS files on
the same repo as you store the code. See code below

```
from __future__ import annotations
from functools import partial
import pathlib

dirpath = pathlib.Path(__file__).absolute().with_suffix(".gds")

def test_gds(component_name: str) -> None:
    component = cells[component_name]()
    test_name = f"fabc_{component_name}"
    difftest(component, test_name=test_name, dirpath=dirpath)

```

"""


import pytest
from pytest_regressions.data_regression import DataRegressionFixture

from gdsfactory.difftest import difftest
from gdsfactory.samples.pdk.fab_c import cells

cell_names = list(cells.keys())


@pytest.fixture(params=cell_names, scope="function")
def component_name(request) -> str:
    return request.param


def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS names, shapes and layers.

    Runs XOR and computes the area.

    """
    component = cells[component_name]()
    test_name = f"fabc_{component_name}"
    difftest(component, test_name=test_name)


def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions in component settings and ports."""
    component = cells[component_name]()
    data_regression.check(component.to_dict())


def test_assert_ports_on_grid(component_name: str) -> None:
    """Ensures all ports are on grid to avoid 1nm gaps."""
    component = cells[component_name]()
    component.assert_ports_on_grid()


if __name__ == "__main__":
    print(cell_names)
    c = cells[cell_names[0]]()
    difftest(c, test_name=f"fabc_{cell_names[0]}")

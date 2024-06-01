from __future__ import annotations

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

from gdsfactory.components import cells
from gdsfactory.config import PATH
from gdsfactory.difftest import difftest
from gdsfactory.serialization import clean_value_json

skip_test = {
    "version_stamp",
    "bbox",
    "component_sequence",
    "extend_ports_list",
    "add_fiber_array_optical_south_electrical_north",
}
cells_to_test = set(cells.keys()) - skip_test


@pytest.fixture(params=cells_to_test)
def component_name(request) -> str:
    return request.param


def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    component = cells[component_name]()
    difftest(component=component, test_name=component_name, dirpath=PATH.gds_ref)


def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    component = cells[component_name]()
    data_regression.check(clean_value_json(component.to_dict()))


if __name__ == "__main__":
    test_gds("pad_rectangular")

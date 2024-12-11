from __future__ import annotations

from typing import Any

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.config import PATH
from gdsfactory.difftest import difftest
from gdsfactory.get_factories import get_cells
from gdsfactory.serialization import clean_value_json

cells = get_cells([gf.components])

skip_test = {
    "version_stamp",
    "bbox",
    "component_sequence",
    "extend_ports_list",
    "add_fiber_array_optical_south_electrical_north",
    "ring_double_pn",
    "pack_doe",
    "pack_doe_grid",
    "text_freetype",
    "grating_coupler_elliptical_lumerical_etch70",
}
cells_to_test = set(cells.keys()) - skip_test


@pytest.fixture(params=cells_to_test)
def component_name(request: pytest.FixtureRequest) -> Any:
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

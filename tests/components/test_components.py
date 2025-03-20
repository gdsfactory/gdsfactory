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
    "component_sequence",
    "extend_ports_list",
    "grating_coupler_elliptical_lumerical_etch70",
    "ring_double_pn",
    "straight_piecewise",
    "text_freetype",
    "version_stamp",
}
cells_to_test = set(cells.keys()) - skip_test


default_container_arguments: dict[str, dict[str, Any]] = dict(
    bbox=dict(component="mmi1x2", layer="SLAB90"),
    pack_doe=dict(doe="mmi1x2", settings=dict(length_mmi=(100, 200))),
    pack_doe_grid=dict(doe="mmi1x2", settings=dict(length_mmi=(100, 200))),
    add_fiber_array_optical_south_electrical_north=dict(
        component=gf.c.straight_heater_metal,
        pad=gf.c.pad,
        grating_coupler=gf.c.grating_coupler_te,
        cross_section_metal="metal_routing",
        pad_pitch=100,
    ),
)


@pytest.fixture(params=cells_to_test)
def component_name(request: pytest.FixtureRequest) -> Any:
    return request.param


def get_component_with_defaults(name: str) -> gf.Component:
    """Get a component, applying default arguments if specified."""
    if name in default_container_arguments:
        return cells[name](**default_container_arguments[name])
    return cells[name]()


def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    component = get_component_with_defaults(component_name)
    difftest(component=component, test_name=component_name, dirpath=PATH.gds_ref)


def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    component = get_component_with_defaults(component_name)
    data_regression.check(clean_value_json(component.to_dict()))


if __name__ == "__main__":
    test_gds("bbox")

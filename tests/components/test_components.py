from __future__ import annotations

import re
from functools import partial
from typing import Any

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.config import PATH
from gdsfactory.difftest import difftest
from gdsfactory.get_factories import get_cells
from gdsfactory.serialization import clean_value_json

cells = get_cells([gf.components])

pad_array_layer = partial(gf.c.pad_array, layer="M1")
pad_array_size = partial(gf.c.pad_array, layer="M1", size=(100, 100))
taper_with_trenches = partial(gf.c.taper, cross_section="rib_with_trenches")

bend_euler_angular_resolution = partial(
    gf.components.bend_euler, radius=10, angle=90, angular_step=20
)

cells.update(
    {
        "pad_array_layer": pad_array_layer,
        "pad_array_size": pad_array_size,
        "taper_with_trenches": taper_with_trenches,
        "bend_euler_angular_resolution": bend_euler_angular_resolution,
    }
)


skip_test = {
    "coupler_bent",
    "component_sequence",
    "extend_ports_list",
    "grating_coupler_elliptical_lumerical_etch70",
    "ring_double_pn",
    "straight_heater_doped_strip",  # TODO: fix this
    "straight_piecewise",
    "text_freetype",
    "version_stamp",
    "die_frame_phix",
    "dbr_tapered",
    "taper_hecken",
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


def _stable_name(name: str) -> str:
    """Drops the counter of unnamed cells, which depends on test order."""
    return re.sub(r"^Unnamed_\d+$", "Unnamed", name)


def _cell_names(component: gf.Component | gf.ComponentAllAngle) -> list[str]:
    """Returns the sorted names of all cells below component."""
    if isinstance(component, gf.Component):
        kcl = component.kcl
        names = {kcl[i].name for i in component.kdb_cell.called_cells()}
    else:
        names = set()
        for inst in [*component.insts, *component.vinsts]:
            names.add(inst.cell.name)
            names.update(_cell_names(inst.cell))
    return sorted({_stable_name(name) for name in names})


def test_cell_names(data_regression: DataRegressionFixture) -> None:
    """Snapshot the name of every cell and of the cells below it.

    A cell name hashes the cell's settings, so the diff of the snapshot lists
    every cell whose name changed.
    """
    snapshot = {}
    for name in sorted(cells_to_test):
        component = get_component_with_defaults(name)
        snapshot[name] = {
            "name": _stable_name(component.name),
            "children": _cell_names(component),
        }
    data_regression.check(snapshot)

from __future__ import annotations

import jsondiff
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.get_factories import get_cells

cells = get_cells([gf.components])

skip_test = {
    "add_fiber_array_optical_south_electrical_north",
    "bbox",
    "component_sequence",
    "cutback_2x2",
    "cutback_bend180circular",
    "cutback_component",
    "delay_snake",
    "delay_snake2",
    "extend_ports_list",
    "pack_doe",
    "pack_doe_grid",
    "spiral_racetrack_fixed_length",
    "straight_heater_metal_simple",
    "via_corner",
    "spiral_racetrack",
    "spiral_racetrack_heater_metal",
    "text_freetype",
}
cells_to_test = set(cells.keys()) - skip_test


@pytest.mark.parametrize("component_type", cells_to_test)
def test_netlists(
    component_type: str,
    data_regression: DataRegressionFixture,
    check: bool = True,
) -> None:
    """Write netlists for hierarchical circuits.

    Checks that both netlists are the same jsondiff does a hierarchical diff.

    Component -> YAML -> Component -> YAML

    then compare YAMLs with pytest regressions
    """
    c = cells[component_type]()
    connection_error_types = {"optical": []}
    n = c.get_netlist(
        allow_multiple=True, connection_error_types=connection_error_types
    )

    if check:
        data_regression.check(n)

    n.pop("warnings", None)
    yaml_str = c.write_netlist(n)
    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist()

    d = jsondiff.diff(n, n2)
    d.pop("warnings", None)
    d.pop("connections", None)
    d.pop("ports", None)
    assert len(d) == 0, d


if __name__ == "__main__":
    component_type = "cdsem_straight"
    component_type = "straight_pn"
    component_type = "splitter_chain"
    component_type = "component_sequence"  # FIXME
    component_type = "cutback_2x2"  # FIXME
    component_type = "pad_array"
    component_type = "terminator"
    component_type = "coupler_adiabatic"
    component_type = "grating_coupler_dual_pol"
    component_type = "greek_cross"
    component_type = "spiral_racetrack_heater_metal"
    component_type = "spiral_racetrack_fixed_length"
    component_type = "via_stack_with_offset"
    component_type = "dbr"
    component_type = "via_corner"
    component_type = "straight_heater_metal_simple"
    component_type = "awg"
    component_type = "grating_coupler_tree"
    component_type = "delay_snake2"  # FIXME
    component_type = "delay_snake"  # FIXME
    component_type = "bbox"
    component_type = "text_freetype"

    connection_error_types = {
        "optical": ["width_mismatch", "shear_angle_mismatch", "orientation_mismatch"]
    }
    connection_error_types = {"optical": []}

    c1 = cells[component_type]()
    c1.show()
    n = c1.get_netlist(
        allow_multiple=True, connection_error_types=connection_error_types
    )
    n.pop("warnings", None)
    yaml_str = c1.write_netlist(n)
    # print(yaml_str)
    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist(allow_multiple=True)
    d = jsondiff.diff(n, n2)
    d.pop("warnings", None)
    c2.show()
    assert len(d) == 0, d

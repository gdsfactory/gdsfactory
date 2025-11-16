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
    "coupler_bend",
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
    "crossing45",
    "seal_ring_segmented",
    "grating_coupler_elliptical_lumerical_etch70",
    "coupler_ring_bend",
    "grating_coupler_array",
    "straight_piecewise",
    "ge_detector_straight_si_contacts",
    "dbr",
    "straight_all_angle",
    "bend_circular_all_angle",
    "bend_euler_all_angle",
    "die_frame_phix",
    "taper_meander",
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
    connection_error_types: dict[str, list[str]] = {"optical": []}
    n = c.get_netlist(
        allow_multiple=True, connection_error_types=connection_error_types
    )

    if check:
        data_regression.check(n)

    # if "warnings" in n:
    #     raise ValueError(n["warnings"])

    n.pop("warnings", None)
    yaml_str = c.write_netlist(n)
    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist()

    d = jsondiff.diff(n, n2)
    d.pop("warnings", None)
    d.pop("connections", None)
    d.pop("ports", None)
    assert len(d) == 0, d

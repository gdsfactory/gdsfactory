from __future__ import annotations

import jsondiff
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.get_factories import get_cells
from gdsfactory.get_netlist import legacy_namer

cells = get_cells([gf.components])

skip_test = {
    "add_fiber_array_optical_south_electrical_north",
    "add_termination",
    "align_wafer",
    "bbox",
    "bend_circular_all_angle",
    "bend_euler_all_angle",
    "component_sequence",
    "coupler_bend",
    "coupler_ring_bend",
    "crossing45",
    "cutback_2x2",
    "cutback_bend180circular",
    "cutback_component",
    "dbr",
    "delay_snake",
    "delay_snake2",
    "die_frame_phix",
    "extend_ports_list",
    "ge_detector_straight_si_contacts",
    "grating_coupler_array",
    "grating_coupler_elliptical_lumerical_etch70",
    "pack_doe",
    "pack_doe_grid",
    "seal_ring_segmented",
    "spiral_racetrack",
    "spiral_racetrack_fixed_length",
    "spiral_racetrack_heater_metal",
    "straight_all_angle",
    "straight_heater_metal_simple",
    "straight_piecewise",
    "taper_meander",
    "text_freetype",
    "via_corner",
}
cells_to_test = sorted(set(cells.keys()) - skip_test)


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
    c1 = cells[component_type]()
    n1 = c1.get_netlist(on_multi_connect="ignore", instance_namer=legacy_namer)

    if check:
        data_regression.check(n1)

    c2 = gf.read.from_yaml(n1)
    n2 = c2.get_netlist(on_multi_connect="ignore", instance_namer=legacy_namer)

    d = jsondiff.diff(n1, n2)
    assert len(d) == 0, d

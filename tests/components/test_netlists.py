from __future__ import annotations

import jsondiff
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.components import cells

skip_test = {
    "add_fiber_array_optical_south_electrical_north",
    "bbox",
    "component_sequence",
    "coupler_bend",
    "cutback_2x2",
    "cutback_bend180circular",
    "cutback_component",
    "delay_snake",
    "delay_snake2",
    "extend_ports_list",
    "fiber_array",
    "grating_coupler_loss_fiber_array",
    "grating_coupler_tree",
    "mzi_lattice",
    "mzi_lattice_mmi",
    "spiral_racetrack",
    "pack_doe",
    "pack_doe_grid",
    "text_freetype",
    "awg",
    "coh_rx_single_pol",
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
    """
    sts[spiral_racetrack_heater_metal] - AssertionError: {'nets': {delete: [3, 2]}}
FAILED tests/components/test_netlists.py::test_netlists[grating_coupler_dual_pol] - AssertionError: {'instances': {'rectangle_S0p3_0p3_LSLA_62b062b0_1535_1535': {'settings': {'port_type': 'electrical'}}, 'rectangle_S0p...rt_type': 'electrical'}}, 'rectangle_S0p3_0p3_LSLA_62b062b0_1535_307': {'settings': {'port_type': 'electrical'}}, ...}}
FAILED tests/components/test_netlists.py::test_netlists[via_corner] - AssertionError: {'nets': {delete: [1, 0]}}
FAILED tests/components/test_netlists.py::test_netlists[pad_array] - AssertionError: {'instances': {'pad_S100_100_LMTOP_BLNo_2fadfa5f_0_0': {'settings': {'port_orientation': 0, 'port_orientations': (180, 90, 0, -90)}}}}
FAILED tests/components/test_netlists.py::test_netlists[pad_array180] - AssertionError: {'instances': {'pad_S100_100_LMTOP_BLNo_2fadfa5f_0_0': {'settings': {'port_orientation': 0, 'port_orientations': (180, 90, 0, -90)}}}}
FAILED tests/components/test_netlists.py::test_netlists[die_with_pads] - AssertionError: {'instances': {'rectangle_S11470_4900_L_392670d4_0_0': {'settings': {'port_type': 'electrical'}}}}
FAILED tests/components/test_netlists.py::test_netlists[switch_tree] - pydantic_core._pydantic_core.ValidationError: 3 validation errors for Netlist
FAILED tests/components/test_netlists.py::test_netlists[greek_cross] - AssertionError: {'nets': ()}
FAILED tests/components/test_netlists.py::test_netlists[spiral_racetrack_fixed_length] - TypeError: 'str' object is not callable
FAILED tests/components/test_netlists.py::test_netlists[pad_array270] - AssertionError: {'instances': {'pad_S100_100_LMTOP_BLNo_2fadfa5f_0_0': {'settings': {'port_orientation': 0, 'port_orientations': (180, 90, 0, -90)}}}}
FAILED tests/components/test_netlists.py::test_netlists[pad_array90] - AssertionError: {'instances': {'pad_S100_100_LMTOP_BLNo_2fadfa5f_0_0': {'settings': {'port_orientation': 0, 'port_orientations': (180, 90, 0, -90)}}}}
FAILED tests/components/test_netlists.py::test_netlists[rectangle_with_slits] - AssertionError: {'instances': {'rectangle_S100_200_LWG__6de38f9c_0_0': {'settings': {'port_type': 'electrical'}}}}
FAILED tests/components/test_netlists.py::test_netlists[via_stack_with_offset] - AssertionError: {'nets': ()}
FAILED tests/components/test_netlists.py::test_netlists[straight_heater_metal_simple] - AssertionError: {'nets': ()}
FAILED tests/components/test_netlists.py::test_netlists[dbr] - pydantic_core._pydantic_core.ValidationError: 1 validation error for Netlist
FAILED tests/components/test_netlists.py::test_netlists[pad_array0]
    """
    # component_type = "mzit"
    component_type = "ring_double"
    component_type = "ring_single_array"
    component_type = "ring_single"
    component_type = "cdsem_straight"
    component_type = "fiber_array"
    component_type = "dbr"
    component_type = "straight_pn"
    component_type = "coupler_bend"  # crashes
    component_type = "splitter_chain"
    component_type = "grating_coupler_loss_fiber_array"
    component_type = "spiral_racetrack"
    component_type = "mzi_lattice_mmi"  # FIXME
    component_type = "straight_heater_meander"  # FIXME: fails
    component_type = "mzi_lattice"  # FIXME
    component_type = "component_sequence"  # FIXME
    component_type = "coupler_bend"  # FIXME
    component_type = "cutback_2x2"  # FIXME
    component_type = "delay_snake"  # FIXME
    component_type = "spiral_racetrack"
    component_type = "pad_array"
    component_type = "terminator"
    component_type = "coupler_adiabatic"
    component_type = "dbr"
    component_type = "grating_coupler_dual_pol"

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

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

    # yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    # c.delete()
    # c2 = gf.read.from_yaml(yaml_str)
    # n2 = c2.get_netlist(
    #     allow_multiple=True, connection_error_types=connection_error_types
    # )

    # n.pop("name")
    # n2.pop("name")
    # n.pop("ports")
    # n2.pop("ports")
    # d = jsondiff.diff(n, n2)
    # d.pop("warnings", None)
    # assert len(d) == 0, d


if __name__ == "__main__":
    import yaml

    from gdsfactory.serialization import convert_tuples_to_lists

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
    component_type = "ring_single"

    connection_error_types = {
        "optical": ["width_mismatch", "shear_angle_mismatch", "orientation_mismatch"]
    }
    connection_error_types = {"optical": []}

    c1 = cells[component_type]()
    c1.show()
    n = c1.get_netlist(
        allow_multiple=True, connection_error_types=connection_error_types
    )
    n = convert_tuples_to_lists(n)
    yaml_str = yaml.dump(n, sort_keys=True)
    c1.delete()
    # print(yaml_str)
    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist(allow_multiple=True)
    d = jsondiff.diff(n, n2)
    d.pop("warnings", None)
    c2.show()
    assert len(d) == 0, d

from __future__ import annotations

import jsondiff
import pytest
from omegaconf import OmegaConf
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.components import cells

skip_test = {
    "version_stamp",
    "bbox",
    "component_sequence",
    "extend_ports_list",
    "add_fiber_array_optical_south_electrical_north",
    "pack_doe",
    "pack_doe_grid",
    "fiber_array",
    "straight_heater_meander",
}
cells_to_test = set(cells.keys()) - skip_test


# @pytest.mark.skip
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
    n = c.get_netlist(allow_multiple=True)
    if check:
        data_regression.check(n)

    n.pop("connections", None)
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c.delete()
    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist()

    n.pop("name")
    n2.pop("name")
    n.pop("ports")
    n2.pop("ports")
    d = jsondiff.diff(n, n2)
    d.pop("warnings", None)
    d.pop("connections", None)
    assert len(d) == 0, d


if __name__ == "__main__":
    # component_type = "mzit"
    component_type = "ring_double"
    component_type = "ring_single_array"
    component_type = "ring_single"
    component_type = "cdsem_straight"
    component_type = "grating_coupler_loss_fiber_array"
    component_type = "fiber_array"
    component_type = "straight_heater_meander"  # FIXME: fails
    component_type = "dbr"
    component_type = "straight_pn"

    connection_error_types = {
        "optical": ["width_mismatch", "shear_angle_mismatch", "orientation_mismatch"]
    }
    connection_error_types = {"optical": []}

    c1 = cells[component_type]()
    c1.show()
    n = c1.get_netlist(
        allow_multiple=True, connection_error_types=connection_error_types
    )
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c1.delete()
    # print(yaml_str)
    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist()
    d = jsondiff.diff(n, n2)
    d.pop("warnings", None)
    assert len(d) == 0, d
    c2.show()

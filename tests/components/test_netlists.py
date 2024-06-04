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

    Component -> netlist -> Component -> netlist

    """
    c = cells[component_type]()
    n = c.get_netlist()
    if check:
        data_regression.check(n)

    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c.delete()
    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist()

    n.pop("name")
    n2.pop("name")
    d = jsondiff.diff(n, n2)
    assert len(d) == 0, d


if __name__ == "__main__":
    # component_type = "mzit"
    component_type = "ring_double"
    component_type = "ring_single_array"
    component_type = "ring_single"
    c1 = cells[component_type]()
    n = c1.get_netlist()
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c1.delete()
    # print(yaml_str)
    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist()
    d = jsondiff.diff(n, n2)
    print(d)
    c2.show()

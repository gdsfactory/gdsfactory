import jsondiff
import pytest
from omegaconf import OmegaConf
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf

components = {
    i: getattr(gf.components, i)
    for i in dir(gf.components)
    if not i.startswith("_") and callable(getattr(gf.components, i))
}

circuit_names = {
    "mzi",
    "ring_single",
    "ring_single_array",
    "ring_double",
    "mzit_lattice",
    "mzit",
    "component_lattice",
}


circuit_names_test = circuit_names - {
    "component_lattice",
    "mzi",
}  # set of component names


@pytest.mark.parametrize("component_type", circuit_names_test)
def test_netlists(
    component_type: str,
    data_regression: DataRegressionFixture,
    check: bool = True,
    component_factory=components,
) -> None:
    """Write netlists for hierarchical circuits.

    Checks that both netlists are the same jsondiff does a hierarchical diff.

    Component -> netlist -> Component -> netlist

    """
    c = component_factory[component_type]()
    n = c.get_netlist()
    if check:
        data_regression.check(n)

    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c2 = gf.read.from_yaml(yaml_str, name=c.name)
    n2 = c2.get_netlist()

    d = jsondiff.diff(n, n2)
    # print(yaml_str)
    # print(d)
    # yaml_str2 = OmegaConf.to_yaml(n2, sort_keys=True)
    # print(yaml_str2)
    assert len(d) == 0, d


def demo_netlist(component_type) -> None:
    c1 = components[component_type]()
    yaml_str = c1.get_netlist_yaml()
    c2 = gf.read.from_yaml(yaml_str, name=c1.name)
    gf.show(c2)


if __name__ == "__main__":
    # c = components["ring_double"]()
    # n = c.get_netlist()

    # c = components["mzi"]()
    # c = components["ring_double"]()

    # gf.clear_connections()
    # print(n.connections)
    # n = c.get_netlist_yaml()
    # print(n)
    # c.show(show_ports=True)

    # c = components["ring_single"]()
    # n.pop("connections")
    # n.pop("placements")

    # component_type = "mzi"
    # component_type = "mzit"
    # component_type = "ring_double"

    component_type = "ring_double"
    component_type = "ring_single"
    component_type = "ring_single_array"
    c1 = components[component_type]()
    n = c1.get_netlist()
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    # print(yaml_str)
    c2 = gf.read.from_yaml(yaml_str, name=c1.name)
    n2 = c2.get_netlist()
    d = jsondiff.diff(n, n2)
    print(d)
    c2.show()

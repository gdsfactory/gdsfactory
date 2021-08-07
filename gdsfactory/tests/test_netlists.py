import itertools as it

import jsondiff
import pytest
from omegaconf import OmegaConf
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.components import circuit_names, component_factory

circuit_names_test = circuit_names - {"component_lattice"}  # set of component names


@pytest.mark.parametrize(
    "component_type,full_settings", it.product(circuit_names_test, [True, False])
)
def test_netlists(
    component_type: str,
    full_settings: bool,
    data_regression: DataRegressionFixture,
    check: bool = True,
) -> None:
    """Write netlists for hierarchical circuits.
    Checks that both netlists are the same
    jsondiff does a hierarchical diff

    Component -> netlist -> Component -> netlist
    """
    c = component_factory[component_type]()
    n = c.get_netlist(full_settings=full_settings)
    if check:
        data_regression.check(n)

    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    # print(yaml_str)
    c2 = gf.component_from_yaml(yaml_str)
    n2 = c2.get_netlist(full_settings=full_settings)

    d = jsondiff.diff(n, n2)
    # yaml_str2 = OmegaConf.to_yaml(n2, sort_keys=True)
    # print(d)
    # print(yaml_str)
    # print(yaml_str2)
    assert len(d) == 0


def demo_netlist(component_type):
    c1 = component_factory[component_type]()
    n = c1.get_netlist()
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c2 = gf.component_from_yaml(yaml_str)
    gf.show(c2)


if __name__ == "__main__":

    # c = component_factory["mzi"]()
    # c = component_factory["ring_double"]()

    # gf.clear_connections()
    # c = component_factory["ring_double"]()
    # n = c.get_netlist()
    # print(n.connections)
    # n = c.get_netlist_yaml()
    # print(n)
    # c.show()

    # c = component_factory["ring_single"]()
    # n.pop("connections")
    # n.pop("placements")

    # component_type = "mzi"
    # component_type = "mzit"
    # component_type = "ring_double"
    # component_type = "ring_single"
    # c1 = component_factory[component_type]()
    # full_settings = True
    # n = c1.get_netlist(full_settings=full_settings)
    # yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    # print(yaml_str)
    # c2 = gf.component_from_yaml(yaml_str)
    # n2 = c2.get_netlist(full_settings=full_settings)
    # d = jsondiff.diff(n, n2)
    # print(d)
    # gf.show(c2)

    test_netlists(
        component_type="ring_single",
        full_settings=True,
        data_regression=None,
        check=False,
    )

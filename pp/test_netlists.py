import itertools as it

import jsondiff
import pytest
from omegaconf import OmegaConf

import pp
from pp.components import circuit_names, component_factory

circuit_names_test = circuit_names - {"component_lattice"}  # set of component names


@pytest.mark.parametrize(
    "component_type,full_settings", it.product(circuit_names_test, [True, False])
)
def test_netlists(component_type, full_settings, data_regression):
    """Write netlists for hierarchical circuits.
    Checks that both netlists are the same
    jsondiff does a hierarchical diff

    Component -> netlist -> Component -> netlist
    """
    c = component_factory[component_type]()
    n = c.get_netlist(full_settings=full_settings)
    data_regression.check(n)

    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c2 = pp.component_from_yaml(yaml_str)
    n2 = c2.get_netlist(full_settings=full_settings)
    d = jsondiff.diff(n, n2)
    assert len(d) == 0


def demo_netlist(component_type):
    c1 = component_factory[component_type]()
    n = c1.get_netlist()
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c2 = pp.component_from_yaml(yaml_str)
    pp.show(c2)


if __name__ == "__main__":

    # c = component_factory["mzi"]()
    # c = component_factory["ring_double"]()

    # pp.clear_connections()
    # c = component_factory["ring_double"]()
    # n = c.get_netlist()
    # print(n.connections)
    # n = c.get_netlist_yaml()
    # print(n)
    # pp.show(c)

    # c = component_factory["ring_single"]()

    component_type = "mzi"
    component_type = "ring_single"
    c1 = component_factory[component_type]()

    full_settings = True

    n = c1.get_netlist(full_settings=full_settings)
    # n.pop("connections")
    # n.pop("placements")
    # pp.clear_cache()
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)

    c2 = pp.component_from_yaml(yaml_str)
    n2 = c2.get_netlist(full_settings=full_settings)

    d = jsondiff.diff(n, n2)
    print(d)
    pp.show(c2)

import pytest
import yaml
from omegaconf import OmegaConf

import pp
from pp.components import _circuits, component_factory

_circuits = _circuits - {"ring_single"}


@pytest.mark.parametrize("component_type", _circuits)
def test_netlists_instances(component_type, data_regression):
    """OmegaConf.save(netlist, "mzi.yml")"""
    pp.clear_cache()
    c = component_factory[component_type]()
    n = c.get_netlist(recursive=True)
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)

    # convert YAML netlist
    d = yaml.load(yaml_str)
    data_regression.check(d)


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
    component_type = "ring_single"
    c1 = component_factory[component_type]()
    n = c1.get_netlist()
    # n.pop("connections")
    pp.clear_cache()
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    print(yaml_str)
    c2 = pp.component_from_yaml(yaml_str)
    pp.show(c2)

import pytest
import yaml
from omegaconf import OmegaConf
import pp
from pp.components import _circuits, component_factory


@pytest.mark.parametrize("component_type", _circuits)
def test_netlists_instances(component_type, data_regression):
    """OmegaConf.save(netlist, "mzi.yml")"""
    c = component_factory[component_type]()
    n = c.get_netlist()
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    d = yaml.load(yaml_str)
    data_regression.check(d)


if __name__ == "__main__":
    # c = component_factory["mzi"]()
    # c = component_factory["ring_single"]()
    # c = component_factory["ring_double"]()

    pp.clear_connections()
    c = component_factory["ring_double"]()
    n = c.get_netlist()
    print(n.connections)
    # n = c.get_netlist_yaml()
    # print(n)
    pp.show(c)

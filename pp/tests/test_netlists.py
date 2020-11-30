import pytest
import yaml
from omegaconf import OmegaConf
import pp
from pp.components import _circuits, component_factory


@pytest.mark.parametrize("component_type", _circuits)
def test_netlists_instances(component_type, data_regression):
    pp.clear_cache()
    c = component_factory[component_type]()
    n = c.get_netlist()
    yaml_str = OmegaConf.to_yaml(n)
    d = yaml.load(yaml_str)
    data_regression.check(d)


if __name__ == "__main__":
    # c = component_factory["mzi"]()
    # c = component_factory["ring_single"]()
    c = component_factory["ring_double"]()
    n = c.get_netlist()
    print(n)
    pp.show(c)

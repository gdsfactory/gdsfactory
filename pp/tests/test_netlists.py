import pytest
import yaml
from omegaconf import OmegaConf
from pp.components import _circuits, component_factory


@pytest.mark.parametrize("component_type", _circuits)
def test_netlists_instances(component_type, data_regression):
    c = component_factory[component_type]()
    n = c.get_netlist()
    yaml_str = OmegaConf.to_yaml(n)
    d = yaml.load(yaml_str)
    data_regression.check(d)


if __name__ == "__main__":
    c = component_factory["mzi"]()
    n = c.get_netlist()
    print(n)

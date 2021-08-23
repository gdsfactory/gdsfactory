"""
FIXME:
    component sequence does not export and import netlists correctly

"""

from omegaconf import OmegaConf
import gdsfactory as gf


if __name__ == "__main__":
    c1 = gf.components.mzi()
    n = c1.get_netlist(full_settings=True)
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)

    c2 = gf.component_from_yaml(yaml_str)

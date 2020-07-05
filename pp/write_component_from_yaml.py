""" write Component from YAML file """

from typing import Union, IO, Any
import pathlib
import io
from omegaconf import OmegaConf

from pp.component import Component
from pp.components import component_type2factory


sample = io.StringIO(
    """
mmi_long:
  component: mmi1x2
  settings:
    width_mmi: 4.5
    length_mmi: 10
  properties:
    x : 100
    y : 100
mmi_short:
  component: mmi1x2
  settings:
    width_mmi: 4.5
    length_mmi: 5
  properties:
    x : 120
    y : 100
"""
)


def read_yaml(file: Union[str, pathlib.Path, IO[Any]]) -> Component:
    """ Loads Components settings from yaml file and writes the GDS into build_directory

    Args:
        file: YAML IO describing DOE

    Returns:
        Component
    """
    c = Component()

    conf = OmegaConf.load(file)
    # cd = OmegaConf.to_container(cd)

    for component_name in conf:
        component_conf = conf[component_name]
        component_type = component_conf["component"]
        component_settings = component_conf["settings"]
        ci = component_type2factory[component_type](**component_settings)
        component_properties = component_conf["properties"]
        for k, v in component_properties.items():
            setattr(ci, k, v)
        ci.name = component_name
        c << ci
    return c


def test_read_yaml():
    pass


if __name__ == "__main__":
    import pp

    c = read_yaml(sample)
    pp.show(c)

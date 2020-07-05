""" write Component from YAML file

"""

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
mmi_short:
  component: mmi1x2
  settings:
    width_mmi: 4.5
    length_mmi: 5
  properties:
    x : 100
    y : 100
"""
)


def component_from_yaml(file: Union[str, pathlib.Path, IO[Any]]) -> Component:
    """ Loads Component settings from YAML file

    Args:
        file: YAML IO describing Component

    Returns:
        Component

    .. code-block:: yaml

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

    """
    c = Component()

    conf = OmegaConf.load(file)

    for component_name in conf:
        component_conf = conf[component_name]
        component_type = component_conf["component"]
        component_settings = component_conf["settings"] or {}
        ci = component_type2factory[component_type](**component_settings)
        component_properties = component_conf["properties"] or {}
        for k, v in component_properties.items():
            setattr(ci, k, v)
        ci.name = component_name
        c << ci
    return c


def test_component_from_yaml():
    c = component_from_yaml(sample)
    assert len(c.get_dependencies()) == 2


if __name__ == "__main__":
    import pp

    c = component_from_yaml(sample)
    assert len(c.get_dependencies()) == 2
    pp.show(c)

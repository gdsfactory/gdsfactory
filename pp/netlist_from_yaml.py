""" write a Component from the YAML netlist

Deprecated! use pp/component_from_yaml instead!

.. code::

             top_arm
        -CP1=       =CP2-
             bot_arm
"""

from typing import Union, IO, Any
import pathlib
import io
from omegaconf import OmegaConf

from pp.component import Component
from pp.components import component_factory as component_factory_default
from pp.netlist_to_gds import netlist_to_component


sample = """

instances:
    CP1:
      component: mmi1x2
      settings:
          width_mmi: 4.5
          length_mmi: 10
    CP2:
        component: mmi1x2
        settings:
            width_mmi: 4.5
            length_mmi: 5
        transformations: mirror_y
    arm_top:
        component: mzi_arm
        settings:
            L0: 10
            DL: 0
    arm_bot:
        component: mzi_arm
        settings:
            L0: 100
            DL: 0
        transformations: mirror_x

connections:
    - [CP1, E0, arm_bot, W0]
    - [arm_bot, E0, CP2, E0]
    - [CP1, E1, arm_top, W0]
    - [arm_top, E0, CP2, E0]

ports_map:
    W0: [CP1, W0]
    E0: [CP2, W0]
    E_TOP_0: [arm_top, E_0]
    E_TOP_1: [arm_top, E_1]
    E_TOP_2: [arm_top, E_2]
    E_TOP_3: [arm_top, E_3]
    E_BOT_0: [arm_bot, E_0]
    E_BOT_1: [arm_bot, E_1]
    E_BOT_2: [arm_bot, E_2]
    E_BOT_3: [arm_bot, E_3]

"""


def netlist_from_yaml(
    yaml: Union[str, pathlib.Path, IO[Any]], component_factory=None,
) -> Component:
    """ Loads Component settings from YAML file, and connections

    Deprecated! use component_from_yaml instead

    Args:
        netlist: YAML IO describing instances, connections and ports_map

    Returns:
        Component

    .. code-block:: yaml

        instances:
            CP1:
              component: mmi1x2
              settings:
                  width_mmi: 4.5
                  length_mmi: 10
            CP2:
                component: mmi1x2
                settings:
                    width_mmi: 4.5
                    length_mmi: 5
                transformations: mirror_y
            arm_top:
                component: mzi_arm
                settings:
                    L0: 10
                    DL: 0
            arm_bot:
                component: mzi_arm
                settings:
                    L0: 100
                    DL: 0
                transformations: mirror_x

        ports_map:
            W0: [CP1, W0]
            E0: [CP2, W0]
            E_TOP_0: [arm_top, E_0]
            E_TOP_1: [arm_top, E_1]
            E_TOP_2: [arm_top, E_2]
            E_TOP_3: [arm_top, E_3]
            E_BOT_0: [arm_bot, E_0]
            E_BOT_1: [arm_bot, E_1]
            E_BOT_2: [arm_bot, E_2]
            E_BOT_3: [arm_bot, E_3]

        connections:
            - [CP1, E0, arm_bot, W0]
            - [arm_bot, E0, CP2, E0]
            - [CP1, E1, arm_top, W0]
            - [arm_top, E0, CP2, E0]

    """

    yaml = io.StringIO(yaml) if isinstance(yaml, str) and "\n" in yaml else yaml
    conf = OmegaConf.load(yaml)
    component_factory = component_factory or component_factory_default

    instances = {}
    for instance_name in conf.instances:
        instance_conf = conf.instances[instance_name]
        component_type = instance_conf["component"]
        component_settings = instance_conf["settings"] or {}
        instance = component_factory[component_type](**component_settings)
        instance_transformations = instance_conf["transformations"] or "None"
        instance_properties = instance_conf["properties"] or {}
        for k, v in instance_properties.items():
            setattr(instance, k, v)
        instance.name = instance_name
        instances[instance_name] = (instance, instance_transformations)

    connections = conf.connections
    ports_map = conf.ports_map
    return netlist_to_component(instances, connections, ports_map)


def test_netlist_from_yaml():
    c = netlist_from_yaml(sample)
    assert len(c.get_dependencies()) == 4
    return c


if __name__ == "__main__":
    import pp

    # c = test_netlist_from_yaml()
    c = netlist_from_yaml(sample)
    pp.show(c)

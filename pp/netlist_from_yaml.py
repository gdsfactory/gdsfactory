""" write a Component from the YAML netlist

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
from pp.components import component_type2factory
from pp.netlist_to_gds import netlist_to_component


sample = io.StringIO(
    """
components:
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
)


def netlist_from_yaml(file: Union[str, pathlib.Path, IO[Any]]) -> Component:
    """ Loads Component settings from YAML file, and connections

    Args:
        file: YAML IO describing components, connections and ports_map

    Returns:
        Component

    .. code-block:: yaml

        components:
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

    conf = OmegaConf.load(file)

    components = {}
    for component_name in conf.components:
        component_conf = conf.components[component_name]
        component_type = component_conf["component"]
        component_settings = component_conf["settings"] or {}
        component = component_type2factory[component_type](**component_settings)
        component_transformations = component_conf["transformations"] or "None"
        component_properties = component_conf["properties"] or {}
        for k, v in component_properties:
            setattr(component, k, v)
        component.name = component_name
        components[component_name] = (component, component_transformations)

    connections = conf.connections
    ports_map = conf.ports_map
    ports_map = {"W0": ("CP1", "W0"), "E0": ("CP2", "W0")}
    return netlist_to_component(components, connections, ports_map)


def test_netlist_from_yaml():
    c = netlist_from_yaml(sample)
    assert len(c.get_dependencies()) == 4


if __name__ == "__main__":
    import pp

    c = netlist_from_yaml(sample)
    pp.show(c)
    # test_netlist_from_yaml()

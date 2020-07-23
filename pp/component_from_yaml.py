""" write Component from YAML file

"""

from typing import Union, IO, Any, Optional
import pathlib
import io
import copy
from omegaconf import OmegaConf

from pp.component import Component
from pp.components import component_type2factory
from pp.routing import link_optical_ports


instances_sample = io.StringIO(
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
"""
)
instances_sample_copy = copy.copy(instances_sample)


placements_sample = io.StringIO(
    """
mmi_long:
    rotation: 180
    x: 100
    y: 100
"""
)


routing_sample = io.StringIO(
    """
mmi_short,E1: mmi_long,E0
"""
)


ports_sample = io.StringIO(
    """
E0: mmi_short,W0
W0: mmi_long,W0
"""
)


valid_placements = ["x", "y", "rotation"]


def component_from_yaml(
    instances: Union[str, pathlib.Path, IO[Any]],
    placements: Optional[Union[str, pathlib.Path, IO[Any]]] = None,
    routing: Optional[Union[str, pathlib.Path, IO[Any]]] = None,
    ports: Optional[Union[str, pathlib.Path, IO[Any]]] = None,
) -> Component:
    """Loads instance settings, placements, routing and ports from YAML

    Args:
        instances: YAML IO describing Component instances
        placements: YAML IO describing component x, y, rotation
        routing: YAML IO describing waveguide routes
        ports: YAML IO describing output component ports

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

    instances_conf = OmegaConf.load(instances)
    placements_conf = OmegaConf.load(placements) if placements else None
    routing_conf = OmegaConf.load(routing) if routing else None
    ports_conf = OmegaConf.load(ports) if ports else None

    instances = {}
    routes = {}

    for instance_name in instances_conf:
        instance_conf = instances_conf[instance_name]
        component_type = instance_conf["component"]
        component_settings = instance_conf["settings"] or {}
        ci = component_type2factory[component_type](**component_settings)
        if placements_conf:
            placement_settings = placements_conf[instance_name] or {}
            for k, v in placement_settings.items():
                if k not in valid_placements:
                    raise ValueError(
                        f"`{k}` not valid placement {valid_placements} for"
                        f" {instance_name}"
                    )
                elif k == "rotation":
                    ci.rotate(v)
                else:
                    setattr(ci, k, v)
        ci.name = instance_name
        instances[instance_name] = c << ci

    if routing_conf:
        for port_in_string, port_out_string in routing_conf.items():
            instance_in_name, port_in_name = port_in_string.split(",")
            instance_out_name, port_out_name = port_out_string.split(",")

            assert (
                instance_in_name in instances
            ), f"{instance_in_name} not in {list(instances.keys())}"
            assert (
                instance_out_name in instances
            ), f"{instance_out_name} not in {list(instances.keys())}"

            instance_in = instances[instance_in_name]
            instance_out = instances[instance_out_name]

            assert port_in_name in instance_in.ports, (
                f"{port_in_name} not in {list(instance_in.ports.keys())} for"
                f" {instance_in_name} "
            )
            assert port_out_name in instance_out.ports, (
                f"{port_out_name} not in {list(instance_out.ports.keys())} for"
                f" {instance_out_name}"
            )

            port_in = instance_in.ports[port_in_name]
            port_out = instance_out.ports[port_out_name]

            route = link_optical_ports([port_in], [port_out])
            c.add(route)
            routes[f"{port_in_string}->{port_out_string}"] = route
    if ports_conf:
        for port_name, instance_comma_port in ports_conf.items():
            instance_name, instance_port_name = instance_comma_port.split(",")
            assert (
                instance_name in instances
            ), f"{instance_name} not in {list(instances.keys())}"
            instance = instances[instance_name]
            assert instance_port_name in instance.ports, (
                f"{instance_port_name} not in {list(instance.ports.keys())} for"
                f" {instance_name} "
            )
            c.add_port(port_name, port=instance.ports[instance_port_name])
    c.instances = instances
    c.routes = routes
    return c


def test_component_from_yaml():
    c = component_from_yaml(instances_sample)
    assert len(c.get_dependencies()) == 2


def test_component_from_yaml_with_routing():
    c = component_from_yaml(
        instances_sample_copy, placements_sample, routing_sample, ports_sample
    )
    assert len(c.get_dependencies()) == 3
    assert len(c.ports) == 2


if __name__ == "__main__":
    import pp

    c = component_from_yaml(
        instances_sample, placements_sample, routing_sample, ports_sample
    )
    # assert len(c.get_dependencies()) == 3
    # test_component_from_yaml()
    # test_component_from_yaml_with_routing()
    # print(c.ports)
    # c = pp.routing.add_io_optical(c)
    pp.show(c)

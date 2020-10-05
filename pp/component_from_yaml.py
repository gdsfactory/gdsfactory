""" write Component from YAML file

"""

from typing import Union, IO, Any
import pathlib
import io
import pytest
from omegaconf import OmegaConf

from pp.component import Component
from pp.components import component_type2factory as component_type2factory_default
from pp.routing import link_optical_ports

valid_placements = ["x", "y", "rotation", "mirror"]
valid_keys = ["instances", "placements", "connections", "ports", "routes"]

sample = """
instances:
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

placements:
    mmi_long:
        rotation: 180
        x: 100
        y: 100

routes:
    mmi_short,E1: mmi_long,E0

ports:
    E0: mmi_short,W0
    W0: mmi_long,W0
"""


sample_connections = """
instances:
    wgw:
      component: waveguide
      settings:
        width: 1
        length: 1
    wgn:
      component: waveguide
      settings:
        width: 0.5
        length: 0.5

connections:
    wgw,E0: wgn,W0

"""


sample_mirror = """
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
    arm_top:
        component: mzi_arm
    arm_bot:
        component: mzi_arm

placements:
    arm_bot:
        mirror: [0, 0, 0, 10]
        rotation: 180
ports:
    W0: CP1,W0
    E0: CP2,W0

connections:
    arm_bot,W0: CP1,E0
    arm_top,W0: CP1,E1
    CP2,E0: arm_bot,E0
    CP2,E0: arm_top,E0
"""


def component_from_yaml(
    yaml: Union[str, pathlib.Path, IO[Any]], component_type2factory=None,
) -> Component:
    """Loads instance settings, placements, routing and ports from YAML

    instances: defines instance names, component and settings
    placements: x, y and rotations
    routes (Optional): defines routes
    ports (Optional): defines ports to expose

    Args:
        yaml: YAML IO describing Component (instances, placements, routing, ports, connections)

    Returns:
        Component

    """
    c = Component()
    yaml = io.StringIO(yaml) if isinstance(yaml, str) and "\n" in yaml else yaml
    component_type2factory = component_type2factory or component_type2factory_default

    conf = OmegaConf.load(yaml)
    for key in conf.keys():
        assert key in valid_keys, f"{key} not in {list(valid_keys)}"

    instances = {}
    routes = {}
    placements_conf = conf.get("placements")
    routing_conf = conf.get("routes")
    ports_conf = conf.get("ports")
    connections_conf = conf.get("connections")

    for instance_name in conf.instances:
        instance_conf = conf.instances[instance_name]
        component_type = instance_conf["component"]
        assert (
            component_type in component_type2factory
        ), f"{component_type} not in {list(component_type2factory.keys())}"
        component_settings = instance_conf["settings"] or {}
        # component_settings.update(cache=False)
        ci = component_type2factory[component_type](**component_settings)
        ci.name = instance_name
        instances[instance_name] = c << ci

        if placements_conf:
            placement_settings = placements_conf[instance_name] or {}
            for k, v in placement_settings.items():
                if k not in valid_placements:
                    raise ValueError(
                        f"`{k}` not valid placement {valid_placements} for"
                        f" {instance_name}"
                    )
                elif k == "rotation":
                    ci.rotate(v, (ci.x, ci.y))
                elif k == "mirror":
                    ci.mirror((v[0], v[1]), (v[2], v[3]))
                else:
                    setattr(ci, k, v)

    if connections_conf:
        for port_src_string, port_dst_string in connections_conf.items():
            instance_src_name, port_src_name = port_src_string.split(",")
            instance_dst_name, port_dst_name = port_dst_string.split(",")

            instance_src_name = instance_src_name.strip()
            instance_dst_name = instance_dst_name.strip()
            port_src_name = port_src_name.strip()
            port_dst_name = port_dst_name.strip()

            assert (
                instance_src_name in instances
            ), f"{instance_src_name} not in {list(instances.keys())}"
            assert (
                instance_dst_name in instances
            ), f"{instance_dst_name} not in {list(instances.keys())}"
            instance_src = instances[instance_src_name]
            instance_dst = instances[instance_dst_name]

            assert port_src_name in instance_src.ports, (
                f"{port_src_name} not in {list(instance_src.ports.keys())} for"
                f" {instance_src_name} "
            )
            assert port_dst_name in instance_dst.ports, (
                f"{port_dst_name} not in {list(instance_dst.ports.keys())} for"
                f" {instance_dst_name}"
            )
            port_dst = instance_dst.ports[port_dst_name]
            instance_src.connect(port=port_src_name, destination=port_dst)

    if routing_conf:
        for port_src_string, port_dst_string in routing_conf.items():
            instance_src_name, port_src_name = port_src_string.split(",")
            instance_dst_name, port_dst_name = port_dst_string.split(",")

            instance_src_name = instance_src_name.strip()
            instance_dst_name = instance_dst_name.strip()
            port_src_name = port_src_name.strip()
            port_dst_name = port_dst_name.strip()

            assert (
                instance_src_name in instances
            ), f"{instance_src_name} not in {list(instances.keys())}"
            assert (
                instance_dst_name in instances
            ), f"{instance_dst_name} not in {list(instances.keys())}"

            instance_in = instances[instance_src_name]
            instance_out = instances[instance_dst_name]

            assert port_src_name in instance_in.ports, (
                f"{port_src_name} not in {list(instance_in.ports.keys())} for"
                f" {instance_src_name} "
            )
            assert port_dst_name in instance_out.ports, (
                f"{port_dst_name} not in {list(instance_out.ports.keys())} for"
                f" {instance_dst_name}"
            )

            port_src = instance_in.ports[port_src_name]
            port_out = instance_out.ports[port_dst_name]

            route = link_optical_ports([port_src], [port_out])
            c.add(route)
            routes[f"{port_src_string}:{port_dst_string}"] = route[0]

    if ports_conf:
        assert hasattr(ports_conf, "items"), f"{ports_conf} needs to be a dict"
        for port_name, instance_comma_port in ports_conf.items():
            instance_name, instance_port_name = instance_comma_port.split(",")
            instance_name = instance_name.strip()
            instance_port_name = instance_port_name.strip()
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


@pytest.fixture(scope="function")
def test_sample():
    c = component_from_yaml(sample)
    assert len(c.get_dependencies()) == 4
    assert len(c.ports) == 2
    return c


@pytest.fixture(scope="function")
def test_connections():
    c = component_from_yaml(sample_connections)
    return c


@pytest.fixture(scope="function")
def test_mirror():
    c = component_from_yaml(sample_mirror)
    return c


@pytest.fixture(scope="function")
def test_netlist_write():
    from pp.components.mzi import mzi

    c = mzi()
    netlist = c.get_netlist()
    # netlist.pop('connections')
    OmegaConf.save(netlist, "mzi.yml")


@pytest.fixture(scope="function")
def test_netlist_read():
    c = component_from_yaml("mzi.yml")
    # print(c.get_netlist().pretty())
    # print(len(c.get_netlist().connections))
    # print((c.get_netlist().connections.pretty()))
    assert len(c.get_dependencies()) == 18
    assert len(c.get_netlist().connections) == 18
    return c


if __name__ == "__main__":
    import pp

    # test_netlist_write()
    # c = test_netlist_read()
    # c = test_mirror()

    c = component_from_yaml(sample)
    pp.show(c)

    # c = component_from_yaml(sample_connections)
    # assert len(c.get_dependencies()) == 3
    # test_component_from_yaml()
    # test_component_from_yaml_with_routing()
    # print(c.ports)
    # c = pp.routing.add_io_optical(c)

""" write Component from YAML file

"""

from typing import Union, IO, Any
import pathlib
import io
from omegaconf import OmegaConf

from pp.component import Component
from pp.components import component_factory as component_factory_default
from pp.routing import route_factory
from pp.routing.connect_bundle import link_ports

valid_placements = ["x", "y", "rotation", "mirror"]
valid_keys = [
    "name",
    "instances",
    "placements",
    "connections",
    "ports",
    "routes",
]

sample_mmis = """
name:
    mmis

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
    optical:
        factory: optical
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
name:
    mirror

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
        settings:
            L0: 30
    arm_bot:
        component: mzi_arm
        settings:
            L0: 30

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
    CP2,E1: arm_top,E0
"""


def component_from_yaml(
    yaml: Union[str, pathlib.Path, IO[Any]],
    component_factory=None,
    route_factory=route_factory,
    **kwargs,
) -> Component:
    """Returns a Component defined from YAML


    Args:
        yaml: YAML IO describing Component (instances, placements, routing, ports, connections)
        component_factory: dict of {factory_name: factory_function}
        route_factory: for routes
        kwargs: cache, pins ... to pass to all factories

    Returns:
        Component

    valid properties:
    name: name of Component
    instances:
        name
        component
        settings
    placements: x, y and rotations
    connections: between instances
    ports (Optional): defines ports to expose
    routes (Optional): defines bundles of routes
    ports (Optional): defines ports to expose

    """
    yaml = io.StringIO(yaml) if isinstance(yaml, str) and "\n" in yaml else yaml
    component_factory = component_factory or component_factory_default

    conf = OmegaConf.load(yaml)
    for key in conf.keys():
        assert key in valid_keys, f"{key} not in {list(valid_keys)}"

    instances = {}
    routes = {}
    name = conf.get("name") or "Unnamed"
    c = Component(name)
    placements_conf = conf.get("placements")
    routes_conf = conf.get("routes")
    ports_conf = conf.get("ports")
    connections_conf = conf.get("connections")

    for instance_name in conf.instances:
        instance_conf = conf.instances[instance_name]
        component_type = instance_conf["component"]
        assert (
            component_type in component_factory
        ), f"{component_type} not in {list(component_factory.keys())}"
        component_settings = instance_conf["settings"] or {}
        component_settings.update(**kwargs)
        ci = component_factory[component_type](**component_settings)
        ref = c << ci
        instances[instance_name] = ref

        if placements_conf:
            placement_settings = placements_conf[instance_name] or {}
            for k, v in placement_settings.items():
                if k not in valid_placements:
                    raise ValueError(
                        f"`{k}` not valid placement {valid_placements} for"
                        f" {instance_name}"
                    )
                elif k == "rotation":
                    ref.rotate(v, (ci.x, ci.y))
                elif k == "mirror":
                    ref.mirror((v[0], v[1]), (v[2], v[3]))
                else:
                    setattr(ref, k, v)

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

    if routes_conf:
        for route_alias in routes_conf:
            route_names = []
            ports1 = []
            ports2 = []
            routes_dict = routes_conf[route_alias]
            if "factory" not in routes_dict:
                raise ValueError(
                    f"`{route_alias}` route needs `factory` : {list(route_factory.keys())}"
                )
            route_type = routes_dict.pop("factory")
            assert (
                route_type in route_factory
            ), f"factory `{route_type}` not in route_factory {list(route_factory.keys())}"
            route_filter = route_factory[route_type]
            route_settings = routes_dict.pop("settings", {})
            for port_src_string, port_dst_string in routes_dict.items():
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

                ports1.append(instance_in.ports[port_src_name])
                ports2.append(instance_out.ports[port_dst_name])
                route_names.append(f"{port_src_string}:{port_dst_string}")
            route = link_ports(
                ports1, ports2, route_filter=route_filter, **route_settings
            )
            for i, r in enumerate(route):
                routes[route_names[i]] = r

            c.add(route)

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


def test_sample():
    c = component_from_yaml(sample_mmis)
    assert len(c.get_dependencies()) == 3
    assert len(c.ports) == 2
    return c


def test_connections():
    c = component_from_yaml(sample_connections)
    # print(len(c.get_dependencies()))
    # print(len(c.ports))
    assert len(c.get_dependencies()) == 2
    assert len(c.ports) == 0
    return c


def test_mirror():
    c = component_from_yaml(sample_mirror)
    # print(len(c.get_dependencies()))
    # print(len(c.ports))
    assert len(c.get_dependencies()) == 3
    assert len(c.ports) == 2
    return c


sample_2x2_connections_problem = """
name:
    connections_2x2_problem

instances:
    mmi_bottom:
      component: mmi2x2
      settings:
            length_mmi: 5
    mmi_top:
      component: mmi2x2
      settings:
            length_mmi: 5

placements:
    mmi_top:
        x: 100
        y: 100

routes:
    optical:
        factory: optical
        mmi_bottom,E0: mmi_top,W0
        mmi_bottom,E1: mmi_top,W1

"""


def test_connections_2x2_problem():
    c = component_from_yaml(sample_2x2_connections_problem, pins=True, cache=False)
    # print(len(c.get_dependencies()))
    # print(len(c.ports))
    assert len(c.get_dependencies()) == 4
    assert len(c.ports) == 0
    return c


sample_2x2_connections_solution = """
name:
    connections_2x2_solution

instances:
    mmi_bottom:
      component: mmi2x2
      settings:
            length_mmi: 5
    mmi_top:
      component: mmi2x2
      settings:
            length_mmi: 5

placements:
    mmi_top:
        x: 100
        y: 100

routes:
    optical:
        factory: optical
        mmi_bottom,E0: mmi_top,W0
        mmi_bottom,E1: mmi_top,W1

"""


def test_connections_2x2_solution():
    c = component_from_yaml(sample_2x2_connections_solution, pins=True, cache=False)
    # print(len(c.get_dependencies()))
    # print(len(c.ports))
    assert len(c.get_dependencies()) == 4
    assert len(c.ports) == 0
    length = c.routes["mmi_bottom,E1:mmi_top,W1"].parent.length
    print(length)
    # assert np.isclose(length, 162.86592653589793)
    return c


sample_order = """

instances:
    bl:
      component: pad
    tl:
      component: pad
    br:
      component: pad
    tr:
      component: pad

placements:
    tl:
        x: 0
        y: 200

    br:
        x: 400
        y: 400

    tr:
        x: 400
        y: 600

routes:
    electrical:
        factory: electrical
        tl,E: tr,W
        bl,E: br,W
    optical:
        factory: optical
        bl,S: br,E

"""


def test_connections_order():
    c = component_from_yaml(sample_order, pins=True, cache=False)
    return c


if __name__ == "__main__":
    import pp

    # c = test_connections_2x2_problem()
    # c = test_connections_2x2_solution()
    # pp.show(c)

    # test_sample()
    # test_connections()
    # test_mirror()
    # c = test_mirror()

    # sample = sample_mmis
    # sample = sample_connections
    sample = sample_mirror
    sample = sample_2x2_connections_solution
    sample = sample_order
    c = component_from_yaml(sample)
    print(c.routes)
    pp.show(c)

    # c = component_from_yaml(sample_connections)
    # assert len(c.get_dependencies()) == 3
    # test_component_from_yaml()
    # test_component_from_yaml_with_routing()
    # print(c.ports)
    # c = pp.routing.add_fiber_array(c)

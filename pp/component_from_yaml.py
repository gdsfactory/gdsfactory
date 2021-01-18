"""Get Component from YAML file."""

import io
import pathlib
from typing import IO, Any, Callable, Dict, List, Optional, Union

import numpy as np
from omegaconf import OmegaConf

from pp.add_pins import _add_instance_label
from pp.component import Component, ComponentReference
from pp.components import component_factory as component_factory_default
from pp.components.extension import move_polar_rad_copy
from pp.routing import link_factory, route_factory

valid_placements = ["x", "y", "dx", "dy", "rotation", "mirror", "port"]
valid_keys = [
    "name",
    "instances",
    "placements",
    "connections",
    "ports",
    "routes",
]

valid_route_keys = ["links", "factory", "settings", "link_factory", "link_settings"]


def place(
    placements_conf: Dict[str, Dict[str, Union[int, float, str]]],
    connections_by_transformed_inst: Dict[str, Dict[str, str]],
    instances: Dict[str, ComponentReference],
    encountered_insts: List[str],
    instance_name: Optional[str] = None,
    all_remaining_insts: Optional[List[str]] = None,
) -> None:
    """Place instance_name with placements_conf config.

    Args:
        placements_conf: Dict of instance_name to placement (x, y, rotation ...)
        connections_by_transformed_inst: Dict of connection attributes, keyed by the name of the instance which should be transformed
        instances: Dict of references
        encountered_insts: list of encountered_instances
        instance_name: instance_name to place
        all_remaining_insts: a list of all the remaining instances which must be placed by this method. Items will be popped from this list as they are placed.
    """
    if not all_remaining_insts:
        return
    if instance_name is None:
        instance_name = all_remaining_insts.pop(0)
    else:
        all_remaining_insts.remove(instance_name)

    if instance_name in encountered_insts:
        encountered_insts.append(instance_name)
        loop_str = " -> ".join(encountered_insts)
        raise ValueError(
            f"circular reference in placement definition for {instance_name}! Loop: {loop_str}"
        )
    encountered_insts.append(instance_name)
    ref = instances[instance_name]

    if instance_name in placements_conf:
        placement_settings = placements_conf[instance_name] or {}
        for k, v in placement_settings.items():
            if k not in valid_placements:
                raise ValueError(
                    f"`{k}` not valid placement {valid_placements} for"
                    f" {instance_name}"
                )
        x = placement_settings.get("x")
        y = placement_settings.get("y")
        dx = placement_settings.get("dx")
        dy = placement_settings.get("dy")
        port = placement_settings.get("port")
        rotation = placement_settings.get("rotation")
        mirror = placement_settings.get("mirror")

        if port:
            a = ref.ports[port]
            ref.x -= a.x
            ref.y -= a.y
        if x:
            if isinstance(x, str):
                if not len(x.split(",")) == 2:
                    raise ValueError(
                        f"You can define x as `x: instaceName,portName` got `x: {x}`"
                    )
                instance_name_ref, port_name = x.split(",")
                if instance_name_ref in all_remaining_insts:
                    place(
                        placements_conf,
                        connections_by_transformed_inst,
                        instances,
                        encountered_insts,
                        instance_name_ref,
                        all_remaining_insts,
                    )
                if instance_name_ref not in instances:
                    raise ValueError(
                        f"instaceName = `{instance_name_ref}` not in {list(instances.keys())}, "
                        f"you can define x as `x: instaceName,portName`, got `x: {x}`"
                    )
                if port_name not in instances[instance_name_ref].ports:
                    raise ValueError(
                        f"portName = `{port_name}` not in {list(instances[instance_name_ref].ports.keys())} "
                        f"for {instance_name_ref}, "
                        f"you can define x as `x: instaceName,portName`, got `x: {x}`"
                    )

                x = instances[instance_name_ref].ports[port_name].x
            ref.x += x
        if y:
            if isinstance(y, str):
                if not len(y.split(",")) == 2:
                    raise ValueError(
                        f"You can define y as `y: instaceName,portName` got `y: {y}`"
                    )
                instance_name_ref, port_name = y.split(",")
                if instance_name_ref in all_remaining_insts:
                    place(
                        placements_conf,
                        connections_by_transformed_inst,
                        instances,
                        encountered_insts,
                        instance_name_ref,
                        all_remaining_insts,
                    )
                if instance_name_ref not in instances:
                    raise ValueError(
                        f"instaceName = `{instance_name_ref}` not in {list(instances.keys())}, "
                        f"you can define y as `y: instaceName,portName`, got `y: {y}`"
                    )
                if port_name not in instances[instance_name_ref].ports:
                    raise ValueError(
                        f"portName = `{port_name}` not in {list(instances[instance_name_ref].ports.keys())} "
                        f"for {instance_name_ref}, "
                        f"you can define y as `y: instaceName,portName`, got `y: {y}`"
                    )

                y = instances[instance_name_ref].ports[port_name].y
            ref.y += y
        if dx:
            ref.x += dx
        if dy:
            ref.y += dy
        if mirror:
            if mirror is True and port:
                port_object = ref.ports[port]
                p1 = port_object.midpoint
                p2 = move_polar_rad_copy(
                    p1, angle=port_object.orientation * np.pi / 180, length=0.2
                )
                ref.reflect(p1=p1, p2=p2)
                # print(port_object.name, p1, p2)
            elif mirror is True:
                if x:
                    ref.reflect_h(x0=x)
                else:
                    ref.reflect_h()
            elif mirror is False:
                pass
            elif isinstance(mirror, str):
                ref.reflect_h(port_name=mirror)
            elif isinstance(mirror, (int, float)):
                ref.reflect_h(x0=mirror)
            else:
                raise ValueError(
                    f"{mirror} can only be a port name {ref.ports.keys()}, a x value or boolean True/False"
                )

        if rotation:
            if port:
                ref.rotate(rotation, center=ref.ports[port])
            else:
                x, y = ref.origin
                ref.rotate(rotation, center=(x, y))
                # ref.rotate(rotation, center=(ref.x, ref.y))

    if instance_name in connections_by_transformed_inst:
        conn_info = connections_by_transformed_inst[instance_name]
        instance_dst_name = conn_info["instance_dst_name"]
        if instance_dst_name in all_remaining_insts:
            place(
                placements_conf,
                connections_by_transformed_inst,
                instances,
                encountered_insts,
                instance_dst_name,
                all_remaining_insts,
            )

        make_connection(instances=instances, **conn_info)
        # placements_conf.pop(instance_name)


def transform_connections_dict(connections_conf: Dict[str, str]) -> Dict[str, Dict]:
    """Returns Dict with source_instance_name key and connection properties."""
    if not connections_conf:
        return {}
    attrs_by_src_inst = {}
    for port_src_string, port_dst_string in connections_conf.items():
        instance_src_name, port_src_name = port_src_string.split(",")
        instance_dst_name, port_dst_name = port_dst_string.split(",")
        attrs_by_src_inst[instance_src_name] = {
            "instance_src_name": instance_src_name,
            "port_src_name": port_src_name,
            "instance_dst_name": instance_dst_name,
            "port_dst_name": port_dst_name,
        }
    return attrs_by_src_inst


def make_connection(
    instance_src_name: str,
    port_src_name: str,
    instance_dst_name: str,
    port_dst_name: str,
    instances: Dict[str, ComponentReference],
) -> None:
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


def component_from_yaml(
    yaml_str: Union[str, pathlib.Path, IO[Any]],
    component_factory: Dict[str, Callable] = None,
    route_factory: Dict[str, Callable] = route_factory,
    link_factory: Dict[str, Callable] = link_factory,
    label_instance_function: Callable = _add_instance_label,
    **kwargs,
) -> Component:
    """Returns a Component defined in YAML file or string.

    Args:
        yaml: YAML IO describing Component file or string (with newlines)
            (instances, placements, routes, ports, connections, names)
        component_factory: dict of {factory_name: factory_function}
        route_factory: for routes
        link_factory: for links
        label_instance_function: to label each instance
        kwargs: cache, pins ... to pass to all factories

    Returns:
        Component

    .. code::

        valid properties:
        name: name of Component
        instances:
            name:
                component:
                settings (Optional)
        placements:
            x: Optional[float, str]  str can be instanceName,portName
            y: Optional[float, str]
            rotation: Optional[float]
            mirror: Optional[bool, float] float is x mirror axis
            port: Optional[str] port anchor
        connections (Optional): between instances
        ports (Optional): defines ports to expose
        routes (Optional): defines bundles of routes
            routeName:
            factory: optical
            links:
                instance1,port1: instance2,port2


    .. code::

        instances:
            mmi_bot:
              component: mmi1x2
              settings:
                width_mmi: 4.5
                length_mmi: 10
            mmi_top:
              component: mmi1x2
              settings:
                width_mmi: 4.5
                length_mmi: 5

        placements:
            mmi_top:
                port: W0
                x: 0
                y: 0
            mmi_bot:
                port: W0
                x: mmi_top,E1
                y: mmi_top,E1
                dx: 30
                dy: -30
        routes:
            optical:
                factory: optical
                links:
                    mmi_top,E0: mmi_bot,W0

    """
    yaml_str = (
        io.StringIO(yaml_str)
        if isinstance(yaml_str, str) and "\n" in yaml_str
        else yaml_str
    )
    component_factory = component_factory or component_factory_default

    conf = OmegaConf.load(yaml_str)  # nicer loader than conf = yaml.safe_load(yaml_str)
    for key in conf.keys():
        assert key in valid_keys, f"{key} not in {list(valid_keys)}"

    instances = {}
    routes = {}
    name = conf.get("name", "Unnamed")
    c = Component(name)
    placements_conf = conf.get("placements")
    routes_conf = conf.get("routes")
    ports_conf = conf.get("ports")
    connections_conf = conf.get("connections")
    instances_dict = conf["instances"]

    for instance_name in instances_dict:
        instance_conf = instances_dict[instance_name]
        component_type = instance_conf["component"]
        assert (
            component_type in component_factory
        ), f"{component_type} not in {list(component_factory.keys())}"
        component_settings = instance_conf["settings"] or {}
        component_settings.update(**kwargs)
        ci = component_factory[component_type](**component_settings)
        ref = c << ci
        instances[instance_name] = ref

    placements_conf = dict() if placements_conf is None else placements_conf

    connections_by_transformed_inst = transform_connections_dict(connections_conf)
    components_to_place = set(placements_conf.keys())
    components_with_placement_conflicts = components_to_place.intersection(
        connections_by_transformed_inst.keys()
    )
    for instance_name in components_with_placement_conflicts:
        placement_settings = placements_conf[instance_name]
        if "x" in placement_settings or "y" in placement_settings:
            print(
                f"YAML defined: ({', '.join(components_with_placement_conflicts)}) "
                + "with both connection and placement. Please use one or the other.",
            )

    all_remaining_insts = list(
        set(placements_conf.keys()).union(set(connections_by_transformed_inst.keys()))
    )

    while all_remaining_insts:
        place(
            placements_conf=placements_conf,
            connections_by_transformed_inst=connections_by_transformed_inst,
            instances=instances,
            encountered_insts=list(),
            all_remaining_insts=all_remaining_insts,
        )

    for instance_name in instances_dict:
        label_instance_function(
            component=c, instance_name=instance_name, reference=instances[instance_name]
        )

    if routes_conf:
        for route_alias in routes_conf:
            route_names = []
            ports1 = []
            ports2 = []
            routes_dict = routes_conf[route_alias]
            if not hasattr(routes_dict, "__items__"):
                print(f"Unvalid syntax for {routes_dict}\n", sample_mmis)
                raise ValueError(f"Unvalid syntax for {routes_dict}")
            for key in routes_dict.keys():
                if key not in valid_route_keys:
                    raise ValueError(
                        f"`{route_alias}` has a key=`{key}` not in valid {valid_route_keys}"
                    )

            if "factory" not in routes_dict:
                raise ValueError(
                    f"`{route_alias}` route needs `factory` : {list(route_factory.keys())}"
                )
            route_type = routes_dict.pop("factory")
            assert isinstance(route_factory, dict), "route_factory needs to be a dict"
            assert (
                route_type in route_factory
            ), f"factory `{route_type}` not in route_factory {list(route_factory.keys())}"
            route_filter = route_factory[route_type]
            route_settings = routes_dict.pop("settings", {})

            link_function_name = routes_dict.pop("link_factory", "link_ports")
            assert (
                link_function_name in link_factory
            ), f"function `{link_function_name}` not in link_factory {list(link_factory.keys())}"
            link_function = link_factory[link_function_name]
            link_settings = routes_dict.pop("link_settings", {})

            if "links" not in routes_dict:
                raise ValueError(
                    f"You need to define links for the `{route_alias}` route"
                )
            links_dict = routes_dict["links"]

            for port_src_string, port_dst_string in links_dict.items():
                # print(port_src_string)

                if ":" in port_src_string:
                    src, src0, src1 = [s.strip() for s in port_src_string.split(":")]
                    dst, dst0, dst1 = [s.strip() for s in port_dst_string.split(":")]
                    instance_src_name, port_src_name = [
                        s.strip() for s in src.split(",")
                    ]
                    instance_dst_name, port_dst_name = [
                        s.strip() for s in dst.split(",")
                    ]

                    src0 = int(src0)
                    src1 = int(src1)
                    dst0 = int(dst0)
                    dst1 = int(dst1)

                    if src1 > src0:
                        ports1names = [
                            f"{port_src_name}{i}" for i in range(src0, src1 + 1, 1)
                        ]
                    else:
                        ports1names = [
                            f"{port_src_name}{i}" for i in range(src0, src1 - 1, -1)
                        ]

                    if dst1 > dst0:
                        ports2names = [
                            f"{port_dst_name}{i}" for i in range(dst0, dst1 + 1, 1)
                        ]
                    else:
                        ports2names = [
                            f"{port_dst_name}{i}" for i in range(dst0, dst1 - 1, -1)
                        ]

                    # print(ports1names)
                    # print(ports2names)

                    assert len(ports1names) == len(ports2names)
                    route_names += [
                        f"{instance_src_name},{i}:{instance_dst_name},{j}"
                        for i, j in zip(ports1names, ports2names)
                    ]

                    instance_src = instances[instance_src_name]
                    instance_dst = instances[instance_dst_name]

                    for port_src_name in ports1names:
                        assert port_src_name in instance_src.ports, (
                            f"{port_src_name} not in {list(instance_src.ports.keys())}"
                            f"for {instance_src_name} "
                        )
                        ports1.append(instance_src.ports[port_src_name])

                    for port_dst_name in ports2names:
                        assert port_dst_name in instance_dst.ports, (
                            f"{port_dst_name} not in {list(instance_dst.ports.keys())}"
                            f"for {instance_dst_name}"
                        )
                        ports2.append(instance_dst.ports[port_dst_name])

                    # print(ports1)
                    # print(ports2)
                    # print(route_names)

                else:
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

                    ports1.append(instance_src.ports[port_src_name])
                    ports2.append(instance_dst.ports[port_dst_name])
                    route_name = f"{port_src_string}:{port_dst_string}"
                    route_names.append(route_name)

            if link_function_name in [
                "link_electrical_waypoints",
                "link_optical_waypoints",
            ]:
                route = link_function(
                    route_filter=route_filter, **route_settings, **link_settings,
                )
                routes[route_name] = route

            else:
                route = link_function(
                    ports1,
                    ports2,
                    route_filter=route_filter,
                    **route_settings,
                    **link_settings,
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
    route_name1:
        factory: optical
        links:
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

#
#        __Lx__
#       |      |
#       Ly     Lyr
#       |      |
#  CP1==|      |==CP2
#       |      |
#       Ly     Lyr
#       |      |
#      DL/2   DL/2
#       |      |
#       |__Lx__|
#

sample_mirror = """
name:
    mzi_with_mirrored_arm

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
            L0: 15

placements:
    arm_bot:
        port: E0
        mirror: True

ports:
    W0: CP1,W0
    E0: CP2,W0

connections:
    arm_bot,W0: CP1,E0
    arm_top,W0: CP1,E1
    CP2,E0: arm_bot,E0
    CP2,E1: arm_top,E0
"""


sample_mirror_simple = """

instances:
    w:
        component: waveguide

    b:
        component: bend_circular

placements:
    b:
        mirror: True
        port: W0

connections:
    b,W0: w,E0

"""


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
    assert len(c.get_dependencies()) == 4
    assert len(c.ports) == 2
    return c


sample_2x2_connections = """
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
            length_mmi: 10

placements:
    mmi_top:
        x: 100
        y: 100

routes:
    optical:
        factory: optical
        links:
            mmi_bottom,E0: mmi_top,W0
            mmi_bottom,E1: mmi_top,W1

"""


def test_connections_2x2():
    c = component_from_yaml(sample_2x2_connections)
    print(len(c.get_dependencies()))
    print(len(c.ports))
    assert len(c.get_dependencies()) == 4
    assert len(c.ports) == 0
    length = c.routes["mmi_bottom,E1:mmi_top,W1"].parent.length
    print(length)
    assert np.isclose(length, 163.91592653589794)
    return c


sample_different_factory = """

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
        settings:
            separation: 240
        links:
            tl,E: tr,W
            bl,E: br,W
    optical:
        factory: optical
        settings:
            bend_radius: 100
        links:
            bl,S: br,E

"""


def test_connections_different_factory():
    c = component_from_yaml(sample_different_factory)
    # print(c.routes["bl,S:br,E"].parent.length)
    assert np.isclose(c.routes["tl,E:tr,W"].parent.length, 700.0)
    assert np.isclose(c.routes["bl,E:br,W"].parent.length, 850.0)
    assert np.isclose(c.routes["bl,S:br,E"].parent.length, 1171.258898038469)
    return c


sample_different_link_factory = """

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
        x: 900
        y: 400

    tr:
        x: 900
        y: 600

routes:
    route1:
        factory: optical
        settings:
            bend_radius: 10
        link_factory: link_ports_path_length_match
        link_settings:
            extra_length: 500
        links:
            tl,E: tr,W
            bl,E: br,W

"""


def test_connections_different_link_factory():
    c = component_from_yaml(sample_different_link_factory)
    # print(c.routes['tl,E:tr,W'].parent.length)
    # print(c.routes['bl,E:br,W'].parent.length)

    length = 1716.2477796076937
    assert np.isclose(c.routes["tl,E:tr,W"].parent.length, length)
    assert np.isclose(c.routes["bl,E:br,W"].parent.length, length)
    return c


sample_waypoints = """

instances:
    t:
      component: pad_array
      settings:
          port_list: ['S']
    b:
      component: pad_array

placements:
    t:
        x: 100
        y: 1000
routes:
    route1:
        factory: optical
        link_factory: link_optical_waypoints
        link_settings:
            way_points:
                - [0,0]
                - [0, 600]
                - [-250, 600]
                - [-250, 1000]
        links:
            t,S5: b,N4
"""


sample_docstring = """
instances:
    mmi_bot:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 10
    mmi_top:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5

placements:
    mmi_top:
        port: W0
        x: 0
        y: 0
    mmi_bot:
        port: W0
        x: mmi_top,E1
        y: mmi_top,E1
        dx: 30
        dy: -30
routes:
    optical:
        factory: optical
        links:
            mmi_top,E0: mmi_bot,W0
"""


sample_regex_connections = """
instances:
    left:
      component: nxn
      settings:
        west: 0
        east: 3
        ysize: 20
    right:
      component: nxn
      settings:
        west: 3
        east: 0
        ysize: 20

placements:
    right:
        x: 20
routes:
    optical:
        factory: optical
        links:
            left,E:0:2: right,W:0:2
"""

sample_regex_connections_backwards = """
instances:
    left:
      component: nxn
      settings:
        west: 0
        east: 3
        ysize: 20
    right:
      component: nxn
      settings:
        west: 3
        east: 0
        ysize: 20

placements:
    right:
        x: 20
routes:
    optical:
        factory: optical
        links:
            left,E:2:0: right,W:2:0
"""


def test_connections_regex():
    c = component_from_yaml(sample_regex_connections)
    route_names = ["left,E0:right,W0", "left,E1:right,W1", "left,E2:right,W2"]

    length = 12.0
    for route_name in route_names:
        print(c.routes[route_name].parent.length)
        assert np.isclose(c.routes[route_name].parent.length, length)
    return c


def test_connections_regex_backwargs():
    c = component_from_yaml(sample_regex_connections_backwards)
    route_names = ["left,E0:right,W0", "left,E1:right,W1", "left,E2:right,W2"]

    length = 12.0
    for route_name in route_names:
        print(c.routes[route_name].parent.length)
        assert np.isclose(c.routes[route_name].parent.length, length)
    return c


def test_connections_waypoints():
    c = component_from_yaml(sample_waypoints)
    # print(c.routes['t,S5:b,N4'].parent.length)

    length = 1241.415926535898
    assert np.isclose(c.routes["t,S5:b,N4"].parent.length, length)
    return c


def test_docstring_sample():
    c = component_from_yaml(sample_docstring)
    route_name = "mmi_top,E0:mmi_bot,W0"
    length = 50.16592653589793
    # print(c.routes[route_name].parent.length)
    assert np.isclose(c.routes[route_name].parent.length, length)
    return c


yaml_fail = """
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
    mmi_short:
        port: W0
        x: mmi_long,E1
        y: mmi_long,E1
    mmi_long:
        port: W0
        x: mmi_short,E1
        y: mmi_short,E1
        dx : 10
        dy: 20
"""


if __name__ == "__main__":
    import pp

    cc = component_from_yaml(yaml_fail)  # this should fail

    # cc = test_connections_regex()
    # cc = component_from_yaml(sample_regex_connections)
    # cc = component_from_yaml(sample_regex_connections_backwards)
    # cc = test_docstring_sample()
    # cc = test_connections()
    # cc = component_from_yaml(sample_mirror_simple)

    # cc = test_connections_2x2()
    # test_sample()
    # cc = test_connections_different_factory()
    # test_connections_different_link_factory()
    # test_connections_waypoints()
    # test_mirror()
    # cc = component_from_yaml(sample_different_link_factory)
    # cc = component_from_yaml(sample_waypoints)
    # cc = test_mirror()
    pp.show(cc)

    # cc = component_from_yaml(sample_connections)
    # assert len(cc.get_dependencies()) == 3
    # test_component_from_yaml()
    # test_component_from_yaml_with_routing()
    # print(cc.ports)
    # cc = pp.routing.add_fiber_array(cc)

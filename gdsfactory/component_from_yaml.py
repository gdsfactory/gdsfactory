"""Returns Component from YAML syntax.

vars:
    length: 3


instances:
    mzi:
        component: mzi_phase_shifter
        settings:
            delta_length: ${defaults.length}
            length_x: 50

    pads:
        component: pad_array
        settings:
            n: 2
            port_names:
                - e4

placements:
    mzi:
        x: 0
    pads:
        y: 200
        x: mzi,cc
ports:
    o1: mzi,o1
    o2: mzi,o2


routes:
    electrical:
        links:
            mzi,etop_e1: pads,e4_0
            mzi,etop_e2: pads,e4_1

        settings:
            layer: [31, 0]
            width: 10
            radius: 10

"""

import functools
import io
import pathlib
import warnings
from typing import IO, Any, Callable, Dict, List, Optional, Union

import numpy as np
import omegaconf
from omegaconf import OmegaConf

from gdsfactory.add_pins import add_instance_label
from gdsfactory.component import Component, ComponentReference
from gdsfactory.components import factory
from gdsfactory.cross_section import cross_section_factory
from gdsfactory.routing.factories import routing_strategy as routing_strategy_factories
from gdsfactory.types import ComponentFactoryDict, CrossSectionFactory, Route

valid_placement_keys = ["x", "y", "dx", "dy", "rotation", "mirror", "port"]


valid_top_level_keys = [
    "name",
    "instances",
    "placements",
    "connections",
    "ports",
    "routes",
    "vars",
]

valid_anchor_point_keywords = [
    "ce",
    "cw",
    "nc",
    "ne",
    "nw",
    "sc",
    "se",
    "sw",
    "center",
    "cc",
]
# refer to an (x,y) Point

valid_anchor_value_keywords = [
    "south",
    "west",
    "east",
    "north",
]
# refer to a singular (x or y) value

valid_anchor_keywords = valid_anchor_point_keywords + valid_anchor_value_keywords
# full set of valid anchor keywords (either referring to points or values)

valid_route_keys = [
    "links",
    "settings",
    "routing_strategy",
]
# Recognized keys within a YAML route definition


def _get_anchor_point_from_name(
    ref: ComponentReference, anchor_name: str
) -> Optional[np.ndarray]:
    if anchor_name in valid_anchor_point_keywords:
        pt = getattr(ref.size_info, anchor_name)
        return pt
    elif anchor_name in ref.ports:
        return ref.ports[anchor_name].position
    else:
        return None


def _get_anchor_value_from_name(
    ref: ComponentReference, anchor_name: str, return_value: str
) -> Optional[float]:
    if anchor_name in valid_anchor_value_keywords:
        v = getattr(ref.size_info, anchor_name)
        return v
    else:
        anchor_point = _get_anchor_point_from_name(ref, anchor_name)
        if anchor_point is None:
            return None
        if return_value == "x":
            return anchor_point[0]
        elif return_value == "y":
            return anchor_point[1]
        else:
            raise ValueError("Expected x or y as return_value.")


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
        connections_by_transformed_inst: Dict of connection attributes.
            keyed by the name of the instance which should be transformed
        instances: Dict of references
        encountered_insts: list of encountered_instances
        instance_name: instance_name to place
        all_remaining_insts: list of all the remaining instances to place
            instances pop from this instrance as they are placed
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
            f"circular reference in placement for {instance_name}! Loop: {loop_str}"
        )
    encountered_insts.append(instance_name)
    ref = instances[instance_name]

    if instance_name in placements_conf:
        placement_settings = placements_conf[instance_name] or {}
        if not isinstance(placement_settings, omegaconf.DictConfig):
            raise ValueError(
                f"Invalid placement {placement_settings} from {valid_placement_keys}"
            )
        for k in placement_settings.keys():
            if k not in valid_placement_keys:
                raise ValueError(f"Invalid placement {k} from {valid_placement_keys}")

        x = placement_settings.get("x")
        y = placement_settings.get("y")
        dx = placement_settings.get("dx")
        dy = placement_settings.get("dy")
        port = placement_settings.get("port")
        rotation = placement_settings.get("rotation")
        mirror = placement_settings.get("mirror")

        if port:
            a = _get_anchor_point_from_name(ref, port)
            if a is None:
                raise ValueError(
                    f"Port {port} is neither a valid port on {ref.parent.name}"
                    " nor a recognized anchor keyword.\n"
                    "Valid ports: \n"
                    f"{list(ref.ports.keys())}. \n"
                    "Valid keywords: \n"
                    f"{valid_anchor_point_keywords}",
                )
            ref.x -= a[0]
            ref.y -= a[1]
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
                        f"instaceName `{instance_name_ref}` not in {list(instances.keys())}, "
                        f"you can define x as `x: instaceName,portName`, got `x: {x}`"
                    )
                if (
                    port_name not in instances[instance_name_ref].ports
                    and port_name not in valid_anchor_keywords
                ):
                    raise ValueError(
                        f"port = `{port_name}` not in {list(instances[instance_name_ref].ports.keys())}"
                        f" or in valid anchors {valid_anchor_keywords} for {instance_name_ref}, "
                        f"you can define x as `x: instaceName,portName`, got `x: {x}`"
                    )

                x = _get_anchor_value_from_name(
                    instances[instance_name_ref], port_name, "x"
                )
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
                        f"{instance_name_ref} not in {list(instances.keys())}, "
                        f"you can define y as `y: instaceName,portName`, got `y: {y}`"
                    )
                if (
                    port_name not in instances[instance_name_ref].ports
                    and port_name not in valid_anchor_keywords
                ):
                    raise ValueError(
                        f"port = {port_name} not in {list(instances[instance_name_ref].ports.keys())} "
                        f"or in valid anchors {valid_anchor_keywords} for {instance_name_ref}, "
                        f"you can define y as `y: instaceName,portName`, got `y: {y}`"
                    )

                y = _get_anchor_value_from_name(
                    instances[instance_name_ref], port_name, "y"
                )
            ref.y += y
        if dx:
            ref.x += dx
        if dy:
            ref.y += dy
        if mirror:
            if mirror is True and port:
                ref.reflect_h(x0=_get_anchor_value_from_name(ref, port, "x"))
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
                    f"{mirror} can only be a port name {ref.ports.keys()}"
                    ", a x value or True/False"
                )

        if rotation:
            if port:
                ref.rotate(rotation, center=_get_anchor_point_from_name(ref, port))
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
        links:
            mmi_short,o2: mmi_long,o1

ports:
    o1: mmi_short,o1
    o2: mmi_long,o2
    o3: mmi_long,o3
"""


def component_from_yaml(
    yaml_str: Union[str, pathlib.Path, IO[Any]],
    component_factory: ComponentFactoryDict = factory,
    routing_strategy: Dict[str, Callable] = routing_strategy_factories,
    cross_section_factory: Dict[str, CrossSectionFactory] = cross_section_factory,
    label_instance_function: Callable = add_instance_label,
    **kwargs,
) -> Component:
    """Returns a Component defined in YAML file or string.

    Args:
        yaml: YAML IO describing Component file or string (with newlines)
            (instances, placements, routes, ports, connections, names)
        component_factory: dict of functions {factory_name: factory_function}
        routing_strategy: for links
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
            library: optical
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
                port: o1
                x: 0
                y: 0
            mmi_bot:
                port: o1
                x: mmi_top,o2
                y: mmi_top,o2
                dx: 30
                dy: -30
        routes:
            optical:
                library: optical
                links:
                    mmi_top,o3: mmi_bot,o1

    """
    yaml_str = (
        io.StringIO(yaml_str)
        if isinstance(yaml_str, str) and "\n" in yaml_str
        else yaml_str
    )

    conf = OmegaConf.load(yaml_str)  # nicer loader than conf = yaml.safe_load(yaml_str)
    for key in conf.keys():
        assert key in valid_top_level_keys, f"{key} not in {list(valid_top_level_keys)}"

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

        settings = instance_conf.get("settings", {})
        settings = OmegaConf.to_container(settings, resolve=True) if settings else {}
        settings.update(**kwargs)

        if "cross_section" in settings:
            name_or_dict = settings["cross_section"]
            if isinstance(name_or_dict, str):
                cross_section = cross_section_factory[name_or_dict]
            elif isinstance(name_or_dict, dict):
                name = name_or_dict.pop("function")
                cross_section = functools.partial(
                    cross_section_factory[name], **name_or_dict
                )
            else:
                raise ValueError(f"invalid type for cross_section={name_or_dict}")
            settings["cross_section"] = cross_section

        ci = component_factory[component_type](**settings)
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
            warnings.warn(
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
            for key in routes_dict.keys():
                if key not in valid_route_keys:
                    raise ValueError(
                        f"`{route_alias}` key=`{key}` not in {valid_route_keys}"
                    )

            settings = routes_dict.pop("settings", {})
            settings = (
                OmegaConf.to_container(settings, resolve=True) if settings else {}
            )
            if "cross_section" in settings:
                name_or_dict = settings["cross_section"]
                if isinstance(name_or_dict, str):
                    cross_section = cross_section_factory[name_or_dict]
                elif isinstance(name_or_dict, dict):
                    name = name_or_dict.pop("function")
                    cross_section = functools.partial(
                        cross_section_factory[name], **name_or_dict
                    )
                else:
                    raise ValueError(f"invalid type for cross_section={name_or_dict}")
                settings["cross_section"] = cross_section
            routing_strategy_name = routes_dict.pop("routing_strategy", "get_bundle")
            if routing_strategy_name not in routing_strategy:
                raise ValueError(
                    f"function `{routing_strategy_name}` not in routing_strategy {list(routing_strategy.keys())}"
                )

            if "links" not in routes_dict:
                raise ValueError(
                    f"You need to define links for the `{route_alias}` route"
                )
            links_dict = routes_dict["links"]

            for port_src_string, port_dst_string in links_dict.items():

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

            routing_function = routing_strategy[routing_strategy_name]
            route_or_route_list = routing_function(
                ports1=ports1,
                ports2=ports2,
                **settings,
            )

            # FIXME, be more consistent
            if isinstance(route_or_route_list, list):
                for route_name, route_dict in zip(route_names, route_or_route_list):
                    c.add(route_dict.references)
                    routes[route_name] = route_dict.length
            elif isinstance(route_or_route_list, Route):
                c.add(route_or_route_list.references)
                routes[route_name] = route_or_route_list.length
            else:
                raise ValueError(f"{route_or_route_list} needs to be a Route or a list")

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
    c.routes = routes
    c.instances = instances
    return c


if __name__ == "__main__":
    for k in factory.keys():
        print(k)
    # c = component_from_yaml(sample_mmis)
    # print(c.get_settings()["info"])

    # from gdsfactory.tests.test_component_from_yaml import yaml_anchor

    # c = component_from_yaml(yaml_anchor)
    # c.show()

    # c = test_connections_regex()
    # c = component_from_yaml(sample_regex_connections)
    # c = component_from_yaml(sample_regex_connections_backwards)
    # c = test_docstring_sample()
    # c = test_connections()
    # c = component_from_yaml(sample_mirror_simple)
    # c = test_connections_2x2()
    # c = test_connections_different_factory()
    # test_connections_different_link_factory()
    # test_connections_waypoints()
    # test_mirror()
    # c = component_from_yaml(sample_different_link_factory)
    # c = test_mirror()
    # c = component_from_yaml(sample_waypoints)
    # c = component_from_yaml(sample_2x2_connections)
    # c = component_from_yaml(sample_mmis)

    # c = component_from_yaml(sample_connections)
    # assert len(c.get_dependencies()) == 3
    # test_component_from_yaml()
    # test_component_from_yaml_with_routing()
    # print(c.ports)
    # c = gf.routing.add_fiber_array(c)

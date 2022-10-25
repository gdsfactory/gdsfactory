"""Returns Component from YAML syntax.

name: myComponent
settings:
    length: 3

info:
    description: just a demo
    polarization: TE
    ...

instances:
    mzi:
        component: mzi_phase_shifter
        settings:
            delta_length: ${settings.length}
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
import importlib
import io
import pathlib
import warnings
from typing import IO, Any, Callable, Dict, List, Optional, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Literal

from gdsfactory.add_pins import add_instance_label
from gdsfactory.cell import cell
from gdsfactory.component import Component, ComponentReference
from gdsfactory.routing.factories import routing_strategy as routing_strategy_factories
from gdsfactory.types import Route

valid_placement_keys = [
    "x",
    "y",
    "xmin",
    "xmax",
    "ymin",
    "ymax",
    "dx",
    "dy",
    "rotation",
    "mirror",
    "port",
]


valid_top_level_keys = [
    "name",
    "instances",
    "placements",
    "connections",
    "ports",
    "routes",
    "settings",
    "info",
    "pdk",
    "warnings",
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
        return getattr(ref.size_info, anchor_name)
    elif anchor_name in ref.ports:
        return ref.ports[anchor_name].center
    else:
        return None


def _get_anchor_value_from_name(
    ref: ComponentReference, anchor_name: str, return_value: str
) -> Optional[float]:
    if anchor_name in valid_anchor_value_keywords:
        return getattr(ref.size_info, anchor_name)
    anchor_point = _get_anchor_point_from_name(ref, anchor_name)
    if anchor_point is None:
        return None
    if return_value == "x":
        return anchor_point[0]
    elif return_value == "y":
        return anchor_point[1]
    else:
        raise ValueError("Expected x or y as return_value.")


def _move_ref(
    x: Union[str, float],
    x_or_y: Literal["x", "y"],
    placements_conf,
    connections_by_transformed_inst,
    instances,
    encountered_insts,
    all_remaining_insts,
) -> float:
    if not isinstance(x, str):
        return x
    if len(x.split(",")) != 2:
        raise ValueError(
            f"You can define {x_or_y} as `{x_or_y}: instaceName,portName` got `{x_or_y}: {x!r}`"
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
            f"{instance_name_ref!r} not in {list(instances.keys())}."
            f" You can define {x_or_y} as `{x_or_y}: instaceName,portName`, got {x_or_y}: {x!r}"
        )
    if (
        port_name not in instances[instance_name_ref].ports
        and port_name not in valid_anchor_keywords
    ):
        ports = list(instances[instance_name_ref].ports.keys())
        raise ValueError(
            f"port = {port_name!r} can be a port_name in {ports}, "
            f"an anchor {valid_anchor_keywords} for {instance_name_ref!r}, "
            f"or `{x_or_y}: instaceName,portName`, got `{x_or_y}: {x!r}`"
        )

    return _get_anchor_value_from_name(instances[instance_name_ref], port_name, x_or_y)


def place(
    placements_conf: Dict[str, Dict[str, Union[int, float, str]]],
    connections_by_transformed_inst: Dict[str, Dict[str, str]],
    instances: Dict[str, ComponentReference],
    encountered_insts: List[str],
    instance_name: Optional[str] = None,
    all_remaining_insts: Optional[List[str]] = None,
) -> None:
    """Place instance_name based on placements_conf config.

    Args:
        placements_conf: Dict of instance_name to placement (x, y, rotation ...).
        connections_by_transformed_inst: Dict of connection attributes.
            keyed by the name of the instance which should be transformed.
        instances: Dict of references.
        encountered_insts: list of encountered_instances.
        instance_name: instance_name to place.
        all_remaining_insts: list of all the remaining instances to place
            instances pop from this instrance as they are placed.

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
    if instance_name not in instances:
        raise ValueError(f"{instance_name!r} not in {list(instances.keys())}")
    ref = instances[instance_name]

    if instance_name in placements_conf:
        placement_settings = placements_conf[instance_name] or {}
        if not isinstance(placement_settings, dict):
            raise ValueError(
                f"Invalid placement {placement_settings} from {valid_placement_keys}"
            )
        for k in placement_settings.keys():
            if k not in valid_placement_keys:
                raise ValueError(f"Invalid placement {k} from {valid_placement_keys}")

        x = placement_settings.get("x")
        xmin = placement_settings.get("xmin")
        xmax = placement_settings.get("xmax")

        y = placement_settings.get("y")
        ymin = placement_settings.get("ymin")
        ymax = placement_settings.get("ymax")

        dx = placement_settings.get("dx")
        dy = placement_settings.get("dy")
        port = placement_settings.get("port")
        rotation = placement_settings.get("rotation")
        mirror = placement_settings.get("mirror")

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
                    f"{mirror!r} can only be a port name {ref.ports.keys()}, "
                    "x value or True/False"
                )

        if port:
            a = _get_anchor_point_from_name(ref, port)
            if a is None:
                raise ValueError(
                    f"Port {port!r} is neither a valid port on {ref.parent.name!r}"
                    " nor a recognized anchor keyword.\n"
                    "Valid ports: \n"
                    f"{list(ref.ports.keys())}. \n"
                    "Valid keywords: \n"
                    f"{valid_anchor_point_keywords}",
                )
            ref.x -= a[0]
            ref.y -= a[1]

        if x is not None:
            ref.x += _move_ref(
                x,
                x_or_y="x",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )

        # print(instance_name, x, xmin, xmax, y, ymin, ymax)
        # print(ymin, y or ymin or ymax)

        if y is not None:
            ref.y += _move_ref(
                y,
                x_or_y="y",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )

        if rotation:
            if port:
                ref.rotate(rotation, center=_get_anchor_point_from_name(ref, port))
            else:
                x, y = ref.origin
                ref.rotate(rotation, center=(x, y))
                # ref.rotate(rotation, center=(ref.x, ref.y))

        if ymin is not None and ymax is not None:
            raise ValueError("You cannot set ymin and ymax")
        elif ymax is not None:
            ref.ymax = _move_ref(
                ymax,
                x_or_y="y",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )
        elif ymin is not None:
            ref.ymin = _move_ref(
                ymin,
                x_or_y="y",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )

        if xmin is not None and xmax is not None:
            raise ValueError("You cannot set xmin and xmax")
        elif xmin is not None:
            ref.xmin = _move_ref(
                xmin,
                x_or_y="x",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )
        elif xmax is not None:
            ref.xmax = _move_ref(
                xmax,
                x_or_y="x",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )
        if dx:
            ref.x += dx

        if dy:
            ref.y += dy

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
    """Connect instance_src_name,port to instance_dst_name,port.

    Args:
        instance_src_name: source instance name.
        port_src_name: from instance_src_name.
        instance_dst_name: destination instance name.
        port_dst_name: from instance_dst_name.
        instances: dict of instances.

    """
    instance_src_name = instance_src_name.strip()
    instance_dst_name = instance_dst_name.strip()
    port_src_name = port_src_name.strip()
    port_dst_name = port_dst_name.strip()

    if instance_src_name not in instances:
        raise ValueError(f"{instance_src_name!r} not in {list(instances.keys())}")
    if instance_dst_name not in instances:
        raise ValueError(f"{instance_dst_name!r} not in {list(instances.keys())}")
    instance_src = instances[instance_src_name]
    instance_dst = instances[instance_dst_name]

    if port_src_name not in instance_src.ports:
        raise ValueError(
            f"{port_src_name} not in {list(instance_src.ports.keys())} for"
            f" {instance_src_name!r} "
        )
    if port_dst_name not in instance_dst.ports:
        raise ValueError(
            f"{port_dst_name!r} not in {list(instance_dst.ports.keys())} for"
            f" {instance_dst_name!r}"
        )
    port_dst = instance_dst.ports[port_dst_name]
    instance_src.connect(port=port_src_name, destination=port_dst)


sample_mmis = """
name: sample_mmis

info:
    polarization: te
    wavelength: 1.55
    description: just a demo on adding metadata

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


def from_yaml(
    yaml_str: Union[str, pathlib.Path, IO[Any], Dict[str, Any], DictConfig],
    routing_strategy: Dict[str, Callable] = routing_strategy_factories,
    label_instance_function: Callable = add_instance_label,
    name: Optional[str] = None,
    prefix: Optional[str] = None,
    **kwargs,
) -> Component:
    """Returns Component from YAML string or file.

    YAML includes instances, placements, routes, ports and connections.

    Args:
        yaml: YAML string or file.
        routing_strategy: for each route.
        label_instance_function: to label each instance.
        name: Optional name.
        prefix: name prefix.
        kwargs: function settings for creating YAML Pcells.

    .. code::

        valid variables:

        name: Optional Component name
        settings: Optional variables
        pdk: overrides
        info: Optional component info
            description: just a demo
            polarization: TE
            ...
        instances:
            name:
                component: (ComponentSpec)
                settings (Optional)
                    length: 10
                    ...
        placements:
            x: Optional[float, str]  str can be instanceName,portName
            y: Optional[float, str]
            rotation: Optional[float]
            mirror: Optional[bool, float] float is x mirror axis
            port: Optional[str] port anchor
        connections (Optional): between instances
        ports (Optional): ports to expose
        routes (Optional): bundles of routes
            routeName:
            library: optical
            links:
                instance1,port1: instance2,port2


    .. code::

        settings:
            length_mmi: 5

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
                length_mmi: ${settings.length_mmi}

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
    if isinstance(yaml_str, (str, pathlib.Path, IO)):
        yaml_str = (
            io.StringIO(yaml_str)
            if isinstance(yaml_str, str) and "\n" in yaml_str
            else yaml_str
        )

        conf = OmegaConf.load(
            yaml_str
        )  # nicer loader than conf = yaml.safe_load(yaml_str)

    else:
        conf = OmegaConf.create(yaml_str)

    for key in conf.keys():
        if key not in valid_top_level_keys:
            raise ValueError(f"{key!r} not in {list(valid_top_level_keys)}")

    settings = conf.get("settings", {})

    for key, value in kwargs.items():
        if key not in settings:
            raise ValueError(f"{key!r} not in {settings.keys()}")
        else:
            conf["settings"][key] = value

    return _from_yaml(
        conf=OmegaConf.to_container(conf, resolve=True),
        routing_strategy=routing_strategy,
        label_instance_function=label_instance_function,
        prefix=prefix or conf.get("name", "Unnamed"),
        name=name,
    )


@cell
def _from_yaml(
    conf,
    routing_strategy: Dict[str, Callable] = routing_strategy_factories,
    label_instance_function: Callable = add_instance_label,
) -> Component:
    """Returns component from YAML decorated with cell for caching and autonaming.

    Args:
        conf: dict.
        routing_strategy: for each route.
        label_instance_function: to label each instance.

    """
    from gdsfactory.pdk import GENERIC, get_active_pdk

    c = Component()
    instances = {}
    routes = {}

    placements_conf = conf.get("placements")
    routes_conf = conf.get("routes")
    ports_conf = conf.get("ports")
    connections_conf = conf.get("connections")
    instances_dict = conf["instances"]
    pdk = conf.get("pdk")
    c.info = conf.get("info", {})

    if pdk and pdk == "generic":
        GENERIC.activate()

    elif pdk:
        module = importlib.import_module(pdk)
        pdk = module.PDK
        if pdk is None:
            raise ValueError(f"'from {pdk} import PDK' failed")
        pdk.activate()

    pdk = get_active_pdk()

    for instance_name in instances_dict:
        instance_conf = instances_dict[instance_name]
        component = instance_conf["component"]
        settings = instance_conf.get("settings", {})
        component_spec = {"component": component, "settings": settings}
        component = pdk.get_component(component_spec)
        ref = c.add_ref(component, alias=instance_name)
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
                "with both connection and placement. Please use one or the other.",
            )

    all_remaining_insts = list(
        set(placements_conf.keys()).union(set(connections_by_transformed_inst.keys()))
    )

    while all_remaining_insts:
        place(
            placements_conf=placements_conf,
            connections_by_transformed_inst=connections_by_transformed_inst,
            instances=instances,
            encountered_insts=[],
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
                        f"{route_alias!r} key={key!r} not in {valid_route_keys}"
                    )

            settings = routes_dict.pop("settings", {})
            routing_strategy_name = routes_dict.pop("routing_strategy", "get_bundle")
            if routing_strategy_name not in routing_strategy:
                routing_strategies = list(routing_strategy.keys())
                raise ValueError(
                    f"{routing_strategy_name!r} is an invalid routing_strategy "
                    f"{routing_strategies}"
                )

            if "links" not in routes_dict:
                raise ValueError(
                    f"You need to define links for the {route_alias!r} route"
                )
            links_dict = routes_dict["links"]

            for port_src_string, port_dst_string in links_dict.items():

                if ":" in port_src_string:
                    src, src0, src1 = (s.strip() for s in port_src_string.split(":"))
                    dst, dst0, dst1 = (s.strip() for s in port_dst_string.split(":"))
                    instance_src_name, port_src_name = (
                        s.strip() for s in src.split(",")
                    )
                    instance_dst_name, port_dst_name = (
                        s.strip() for s in dst.split(",")
                    )

                    src0 = int(src0)
                    src1 = int(src1)
                    dst0 = int(dst0)
                    dst1 = int(dst1)

                    if src1 > src0:
                        ports1names = [
                            f"{port_src_name}{i}" for i in range(src0, src1 + 1)
                        ]
                    else:
                        ports1names = [
                            f"{port_src_name}{i}" for i in range(src0, src1 - 1, -1)
                        ]

                    if dst1 > dst0:
                        ports2names = [
                            f"{port_dst_name}{i}" for i in range(dst0, dst1 + 1)
                        ]
                    else:
                        ports2names = [
                            f"{port_dst_name}{i}" for i in range(dst0, dst1 - 1, -1)
                        ]

                    if len(ports1names) != len(ports2names):
                        raise ValueError(f"{ports1names} different from {ports2names}")

                    route_names += [
                        f"{instance_src_name},{i}:{instance_dst_name},{j}"
                        for i, j in zip(ports1names, ports2names)
                    ]

                    instance_src = instances[instance_src_name]
                    instance_dst = instances[instance_dst_name]

                    for port_src_name in ports1names:
                        if port_src_name not in instance_src.ports:
                            raise ValueError(
                                f"{port_src_name!r} not in {list(instance_src.ports.keys())}"
                                f"for {instance_src_name!r} "
                            )
                        ports1.append(instance_src.ports[port_src_name])

                    for port_dst_name in ports2names:
                        if port_dst_name not in instance_dst.ports:
                            raise ValueError(
                                f"{port_dst_name!r} not in {list(instance_dst.ports.keys())}"
                                f"for {instance_dst_name!r}"
                            )
                        ports2.append(instance_dst.ports[port_dst_name])

                else:
                    instance_src_name, port_src_name = port_src_string.split(",")
                    instance_dst_name, port_dst_name = port_dst_string.split(",")

                    instance_src_name = instance_src_name.strip()
                    instance_dst_name = instance_dst_name.strip()
                    port_src_name = port_src_name.strip()
                    port_dst_name = port_dst_name.strip()

                    if instance_src_name not in instances:
                        raise ValueError(
                            f"{instance_src_name!r} not in {list(instances.keys())}"
                        )
                    if instance_dst_name not in instances:
                        raise ValueError(
                            f"{instance_dst_name!r} not in {list(instances.keys())}"
                        )

                    instance_src = instances[instance_src_name]
                    instance_dst = instances[instance_dst_name]

                    if port_src_name not in instance_src.ports:
                        raise ValueError(
                            f"{port_src_name!r} not in {list(instance_src.ports.keys())} for"
                            f" {instance_src_name!r} "
                        )

                    if port_dst_name not in instance_dst.ports:
                        raise ValueError(
                            f"{port_dst_name!r} not in {list(instance_dst.ports.keys())} for"
                            f" {instance_dst_name!r}"
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
        if not hasattr(ports_conf, "items"):
            raise ValueError(f"{ports_conf} needs to be a dict")
        for port_name, instance_comma_port in ports_conf.items():
            if "," in instance_comma_port:
                instance_name, instance_port_name = instance_comma_port.split(",")
                instance_name = instance_name.strip()
                instance_port_name = instance_port_name.strip()
                if instance_name not in instances:
                    raise ValueError(
                        f"{instance_name!r} not in {list(instances.keys())}"
                    )
                instance = instances[instance_name]

                if instance_port_name not in instance.ports:
                    raise ValueError(
                        f"{instance_port_name!r} not in {list(instance.ports.keys())} for"
                        f" {instance_name!r} "
                    )
                c.add_port(port_name, port=instance.ports[instance_port_name])
            else:
                c.add_port(**instance_comma_port)

    c.routes = routes
    c.info["instances"] = list(instances.keys())
    return c


sample_pdk = """

pdk: ubcpdk

info:
    polarization: te
    wavelength: 1.55
    description: mzi for ubcpdk


instances:
    yr:
      component: y_splitter
    yl:
      component: y_splitter

placements:
    yr:
        rotation: 180
        x: 100
        y: 100

routes:
    route_top:
        links:
            yl,opt2: yr,opt3
    route_bot:
        links:
            yl,opt3: yr,opt2
        routing_strategy: get_bundle_from_steps


ports:
    o1: yl,opt1
    o2: yr,opt2
    o3: yr,opt3

"""

sample_pdk_mzi = """
name: mzi
pdk: ubcpdk

info:
    polarization: te
    wavelength: 1.55
    description: mzi for ubcpdk

instances:
    yr:
      component: y_splitter
    yl:
      component: y_splitter

placements:
    yr:
        rotation: 180
        x: 100
        y: 0

routes:
    route_top:
        links:
            yl,opt2: yr,opt3
    route_bot:
        links:
            yl,opt3: yr,opt2
        routing_strategy: get_bundle_from_steps
        settings:
          steps: [dx: 30, dy: -40, dx: 20]

ports:
    o1: yl,opt1
    o2: yr,opt2
    o3: yr,opt3

"""


sample_pdk_mzi_settings = """
name: mzi

pdk: ubcpdk

settings:
   dy: -70

info:
    polarization: te
    wavelength: 1.55
    description: mzi for ubcpdk

instances:
    yr:
      component: ebeam_y_1550
    yl:
      component: ebeam_y_1550

placements:
    yr:
        rotation: 180
        x: 100
        y: 0

routes:
    route_top:
        links:
            yl,opt2: yr,opt3
        settings:
            cross_section: strip
    route_bot:
        links:
            yl,opt3: yr,opt2
        routing_strategy: get_bundle_from_steps
        settings:
          steps: [dx: 30, dy: '${settings.dy}', dx: 20]
          cross_section: strip


ports:
    o1: yl,opt1
    o2: yr,opt1
"""


sample_pdk_mzi_lattice = """
name: lattice_filter
pdk: ubcpdk

instances:
    mzi1:
      component: mzi.icyaml

    mzi2:
      component: mzi.icyaml

"""


sample_yaml_xmin = """
name: mask_compact

instances:
    mmi1x2_sweep_pack:
       component: pack_doe
       settings:
         component: mmi1x2
         length_mmi: [2, 100]
         width_mmi: [4, 10]
       pack:
         do_permutations: True
         spacing: 100

    mzi_sweep:
       component: pack_doe
       settings:
         component: mzi
         delta_length: [10, 100]
       pack:
         do_permutations: True
         spacing: 100

placements:
    mmi1x2_sweep_pack:
        xmin: -10

    mzi_sweep:
        xmin: mmi1x2_sweep_pack,east

"""


sample_doe = """
name: mask_compact
pdk: ubcpdk

instances:
  rings:
    component: pack_doe
    settings:
      doe: ring_single
      settings:
        radius: [30, 50, 20, 40]
        length_x: [1, 2, 3]
      do_permutations: True
      function:
        function: add_fiber_array
        settings:
            fanout_length: 200


  mzis:
    component: pack_doe_grid
    settings:
      doe: mzi
      settings:
        delta_length: [10, 100]
      do_permutations: True
      spacing: [10, 10]
      function: add_fiber_array

placements:
  rings:
    xmin: 50

  mzis:
    xmin: rings,east

"""


sample_add_gratings = """
name: sample_add_gratings
pdk: ubcpdk
instances:
  ring_te:
    component:
        component: add_fiber_array
        settings:
            component: ring_single

"""

sample_add_gratings_doe = """
name: sample_add_gratings_doe
pdk: ubcpdk
instances:
  ring_te:
    component:
        component: pack_doe
        settings:
            component: add_fiber_array
            settings:
                component: ring_single

"""


sample_rotation_hacky = """
name: sample_rotation

instances:
  r1:
    component: rectangle
    settings:
        size: [4, 2]
  r2:
    component: rectangle
    settings:
        size: [2, 4]

placements:
    r1:
        xmin: 0
        ymin: 0
    r2:
        rotation: 90
        xmin: r1,west
        ymin: 0

"""

sample_rotation = """
name: sample_rotation

instances:
  r1:
    component: rectangle
    settings:
        size: [4, 2]
  r2:
    component: rectangle
    settings:
        size: [2, 4]

placements:
    r1:
        xmin: 0
        ymin: 0
    r2:
        rotation: -90
        xmin: r1,east
        ymin: 0

"""

sample2 = """

name: sample_different_factory2

instances:
    tl:
      component: pad
    tr:
      component: pad

    mzi:
      component: mzi_phase_shifter_top_heater_metal

placements:
    mzi:
        ymax: tl,south
        dy: -100
    tl:
        x: mzi,west
        y: mzi,north
        dy: 100
    tr:
        x: mzi,west
        dx: 200
        y: mzi,north
        dy: 100

routes:
    electrical1:
        routing_strategy: get_bundle
        settings:
            separation: 20
            layer: [31, 0]
            width: 10

        links:
            mzi,e2: tr,e1

    electrical2:
        routing_strategy: get_bundle
        settings:
            separation: 20
            layer: [31, 0]
            width: 10

        links:
            mzi,e1: tl,e1


"""

sample_mirror = """
name: sample_mirror
instances:
    mmi1:
      component: mmi1x2

    mmi2:
      component: mmi1x2

placements:
    mmi1:
        xmax: 0

    mmi2:
        xmin: mmi1,east
        mirror: True

"""

sample_doe_function = """
name: mask_compact

instances:
  rings:
    component: pack_doe
    settings:
      doe: ring_single
      settings:
        radius: [30, 50, 20, 40]
        length_x: [1, 2, 3]
      do_permutations: True
      function:
        function: add_fiber_array
        settings:
            fanout_length: 200


  mzis:
    component: pack_doe_grid
    settings:
      doe: mzi
      settings:
        delta_length: [10, 100]
      do_permutations: True
      spacing: [10, 10]
      function: add_fiber_array

placements:
  rings:
    xmin: 50

  mzis:
    xmin: rings,east

"""


if __name__ == "__main__":
    c = from_yaml(sample_doe_function)
    c = from_yaml(sample_mmis)
    n = c.get_netlist()
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c2 = from_yaml(yaml_str)
    n2 = c2.get_netlist()
    c2.show()

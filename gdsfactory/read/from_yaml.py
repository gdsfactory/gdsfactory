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

from __future__ import annotations

import importlib
import io
import pathlib
import warnings
from collections.abc import Callable
from functools import partial
from typing import IO, Any, Literal

import kfactory as kf
from omegaconf import DictConfig, OmegaConf

from gdsfactory.add_pins import add_instance_label
from gdsfactory.component import Component, Instance
from gdsfactory.serialization import clean_value_json

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
    "na",
    "nb",
    "dax",
    "dbx",
    "day",
    "dby",
]


valid_top_level_keys = [
    "name",
    "instances",
    "placements",
    "connections",
    "nets",
    "ports",
    "routes",
    "settings",
    "info",
    "pdk",
    "warnings",
    "schema",
    "schema_version",
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
    ref: Instance, anchor_name: str
) -> tuple[float, float] | None:
    if anchor_name in valid_anchor_point_keywords:
        return getattr(ref.dsize_info, anchor_name)
    elif anchor_name in ref.ports:
        return ref.ports[anchor_name].dcenter
    else:
        return None


def _get_anchor_value_from_name(
    ref: Instance, anchor_name: str, return_value: str
) -> float | None:
    if anchor_name in valid_anchor_value_keywords:
        return getattr(ref.dsize_info, anchor_name)
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
    x: str | float,
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
            f"You can define {x_or_y} as `{x_or_y}: instanceName,portName` got `{x_or_y}: {x!r}`"
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
            f" You can define {x_or_y} as `{x_or_y}: instanceName,portName`, got {x_or_y}: {x!r}"
        )
    if (
        port_name not in instances[instance_name_ref].ports
        and port_name not in valid_anchor_keywords
    ):
        ports = [p.name for p in instances[instance_name_ref].ports]
        raise ValueError(
            f"port = {port_name!r} can be a port_name in {ports}, "
            f"an anchor {valid_anchor_keywords} for {instance_name_ref!r}, "
            f"or `{x_or_y}: instanceName,portName`, got `{x_or_y}: {x!r}`"
        )

    return _get_anchor_value_from_name(instances[instance_name_ref], port_name, x_or_y)


def _parse_maybe_arrayed_instance(inst_spec: str) -> tuple:
    if inst_spec.count("<") > 1:
        raise ValueError(
            f"Too many angle brackets (<) in instance specification '{inst_spec}'. Array ref indices should end with <ia.ib>, and otherwise this character should be avoided."
        )
    if "<" in inst_spec and inst_spec.endswith(">"):
        inst_name, array_spec = inst_spec.split("<")
        array_spec = array_spec[:-1]
        if "." not in array_spec:
            raise ValueError(
                f"Array specifier should contain a '.' and be of the format my_ref<ia.ib>. Got {inst_spec}"
            )
        if array_spec.count(".") > 1:
            raise ValueError(
                f"Too many periods (.) in array specifier. Array specifier should be of the format my_ref<ia.ib>. Got {inst_spec}"
            )
        ia, ib = array_spec.split(".")
        try:
            ia = int(ia)
        except ValueError:
            raise ValueError(
                f"When parsing array reference specifier '{inst_spec}', got a non-integer index '{ia}'"
            )
        try:
            ib = int(ib)
        except ValueError:
            raise ValueError(
                f"When parsing array reference specifier '{inst_spec}', got a non-integer index '{ib}'"
            )
        return inst_name, ia, ib
    # in the non-arrayed case, return none for the indices
    return inst_spec, None, None


def place(
    placements_conf: dict[str, dict[str, int | float | str]],
    connections_by_transformed_inst: dict[str, dict[str, str]],
    instances: dict[str, Instance],
    encountered_insts: list[str],
    instance_name: str | None = None,
    all_remaining_insts: list[str] | None = None,
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
            instances pop from this instance as they are placed.

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

        if rotation:
            if port:
                ref.drotate(rotation, center=_get_anchor_point_from_name(ref, port))
            else:
                ref.drotate(rotation)

        if mirror:
            if mirror is True and port:
                ref.dmirror_x(x=_get_anchor_value_from_name(ref, port, "x"))
            elif mirror is True:
                ref.dcplx_trans *= kf.kdb.DCplxTrans(1, 0, True, 0, 0)
            elif mirror is False:
                pass
            elif isinstance(mirror, str):
                x_mirror = ref.ports[mirror].dx
                ref.dmirror_x(x_mirror)
            elif isinstance(mirror, int | float):
                ref.dmirror_x(x=ref.dx)
            else:
                port_names = [port.name for port in ref.ports]
                raise ValueError(
                    f"{mirror!r} can only be a port name {port_names}, "
                    "x value or True/False"
                )

        if port:
            a = _get_anchor_point_from_name(ref, port)
            if a is None:
                port_names = [port.name for port in ref.ports]
                raise ValueError(
                    f"Port {port!r} is neither a valid port on {ref.parent.name!r}"
                    " nor a recognized anchor keyword.\n"
                    "Valid ports: \n"
                    f"{port_names}. \n"
                    "Valid keywords: \n"
                    f"{valid_anchor_point_keywords}",
                )
            ref.dx -= a[0]
            ref.dy -= a[1]

        if x is not None:
            ref.dx += _move_ref(
                x,
                x_or_y="x",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )

        if y is not None:
            ref.dy += _move_ref(
                y,
                x_or_y="y",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )

        if ymin is not None and ymax is not None:
            raise ValueError("You cannot set ymin and ymax")
        elif ymax is not None:
            ref.dymax = _move_ref(
                ymax,
                x_or_y="y",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )
        elif ymin is not None:
            ref.dymin = _move_ref(
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
            ref.dxmin = _move_ref(
                xmin,
                x_or_y="x",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )
        elif xmax is not None:
            ref.dxmax = _move_ref(
                xmax,
                x_or_y="x",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )
        if dx:
            ref.dx += dx

        if dy:
            ref.dy += dy

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


def transform_connections_dict(connections_conf: dict[str, str]) -> dict[str, dict]:
    """Returns Dict with source_instance_name key and connection properties."""
    if not connections_conf:
        return {}
    attrs_by_src_inst = {}
    for port_src_string, port_dst_string in connections_conf.items():
        instance_src_name, port_src_name = port_src_string.split(",")
        instance_dst_name, port_dst_name = port_dst_string.split(",")
        instance_src_name, src_ia, src_ib = _parse_maybe_arrayed_instance(
            instance_src_name
        )
        instance_dst_name, dst_ia, dst_ib = _parse_maybe_arrayed_instance(
            instance_dst_name
        )
        attrs_by_src_inst[instance_src_name] = {
            "instance_src_name": instance_src_name,
            "port_src_name": port_src_name,
            "instance_dst_name": instance_dst_name,
            "port_dst_name": port_dst_name,
        }
        src_dict = attrs_by_src_inst[instance_src_name]
        if src_ia is not None:
            src_dict["src_ia"] = src_ia
            src_dict["src_ib"] = src_ib
        if dst_ia is not None:
            src_dict["dst_ia"] = dst_ia
            src_dict["dst_ib"] = dst_ib
    return attrs_by_src_inst


def make_connection(
    instance_src_name: str,
    port_src_name: str,
    instance_dst_name: str,
    port_dst_name: str,
    instances: dict[str, Instance],
    src_ia: int | None = None,
    src_ib: int | None = None,
    dst_ia: int | None = None,
    dst_ib: int | None = None,
) -> None:
    """Connect instance_src_name,port to instance_dst_name,port.

    Args:
        instance_src_name: source instance name.
        port_src_name: from instance_src_name.
        instance_dst_name: destination instance name.
        port_dst_name: from instance_dst_name.
        instances: dict of instances.
        src_ia: the a-index of the source instance, if it is an arrayed instance
        src_ib: the b-index of the source instance, if it is an arrayed instance
        dst_ia: the a-index of the destination instance, if it is an arrayed instance
        dst_ib: the b-index of the destination instance, if it is an arrayed instance

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
        instance_src_port_names = [p.name for p in instance_src.ports]
        raise ValueError(
            f"{port_src_name!r} not in {instance_src_port_names} for"
            f" {instance_src_name!r} "
        )
    if port_dst_name not in instance_dst.ports:
        instance_dst_port_names = [p.name for p in instance_dst.ports]
        raise ValueError(
            f"{port_dst_name!r} not in {instance_dst_port_names} for"
            f" {instance_dst_name!r}"
        )

    if src_ia is None:
        src_port = instance_src.ports[port_src_name]
    else:
        src_port = instance_src.ports[(port_src_name, src_ia, src_ib)]

    if dst_ia is None:
        dst_port = instance_dst.ports[port_dst_name]
    else:
        dst_port = instance_dst.ports[(port_dst_name, dst_ia, dst_ib)]

    instance_src.connect(port=src_port, other=dst_port, use_mirror=True, mirror=False)


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


def cell_from_yaml(
    yaml_str: str | pathlib.Path | IO[Any] | dict[str, Any] | DictConfig,
    routing_strategy: dict[str, Callable] | None = None,
    label_instance_function: Callable = add_instance_label,
    name: str | None = None,
    prefix: str | None = None,
    **kwargs,
) -> Callable:
    """Returns Component factory from YAML string or file.

    YAML includes instances, placements, routes, ports and connections.

    Args:
        yaml_str: YAML string or file.
        routing_strategy: for each route.
        label_instance_function: to label each instance.
        name: Optional name.
        prefix: name prefix.
        kwargs: function settings for creating YAML PCells.

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
            x: float, str | None  str can be instanceName,portName
            y: float, str | None
            rotation: float | None
            mirror: bool, float | None float is x mirror axis
            port: str | None port anchor
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
    return partial(
        from_yaml,
        yaml_str=yaml_str,
        routing_strategy=routing_strategy,
        label_instance_function=label_instance_function,
        name=name,
        prefix=prefix,
        **kwargs,
    )


def from_yaml(
    yaml_str: str | pathlib.Path | IO[Any] | dict[str, Any] | DictConfig,
    routing_strategy: dict[str, Callable] | None = None,
    label_instance_function: Callable = add_instance_label,
    name: str | None = None,
    **kwargs,
) -> Component:
    """Returns Component from YAML string or file.

    YAML includes instances, placements, routes, ports and connections.

    Args:
        yaml_str: YAML string or file.
        routing_strategy: for each route.
        label_instance_function: to label each instance.
        name: Optional name.
        kwargs: function settings for creating YAML PCells.

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
            x: float, str | None  str can be instanceName,portName
            y: float, str | None
            rotation: float | None
            mirror: bool, float | None float is x mirror axis
            port: str | None port anchor
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
    from gdsfactory.generic_tech import get_generic_pdk
    from gdsfactory.pdk import get_active_pdk, get_routing_strategies

    if routing_strategy is None:
        routing_strategy = get_routing_strategies()
    if isinstance(yaml_str, str | pathlib.Path | IO):
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
    mode = kwargs.pop("mode") if "mode" in kwargs else "layout"
    for key, value in kwargs.items():
        if key not in settings:
            raise ValueError(f"{key!r} not in {settings.keys()}")
        else:
            conf["settings"][key] = value

    conf = OmegaConf.to_container(conf, resolve=True)
    name = conf.get("name", name)

    c = Component(name)
    instances = {}
    routes = {}

    placements_conf = conf.get("placements")
    routes_conf = conf.get("routes")
    ports_conf = conf.get("ports")
    connections_conf = conf.get("connections")
    instances_dict = conf["instances"]
    pdk = conf.get("pdk")
    info = conf.get("info", {})

    for key, value in info.items():
        c.info[key] = value

    if pdk and pdk == "generic":
        GENERIC = get_generic_pdk()
        GENERIC.activate()

    elif pdk:
        module = importlib.import_module(pdk)
        pdk = module.PDK
        if pdk is None:
            raise ValueError(f"'from {pdk} import PDK' failed")
        pdk.activate()

    pdk = get_active_pdk()
    if mode == "layout":
        component_getter = pdk.get_component
    elif mode == "schematic":
        component_getter = pdk.get_symbol
    else:
        raise ValueError(
            f"{mode} is not a recognized mode. Please choose 'layout' or 'schematic'"
        )

    for instance_name in instances_dict:
        instance_conf = instances_dict[instance_name]
        component = instance_conf["component"]
        settings = instance_conf.get("settings", {})
        settings = clean_value_json(settings)
        component_spec = {"component": component, "settings": settings}
        component = component_getter(component_spec)
        ref = c.add_ref(
            component,
            name=instance_name,
            rows=instance_conf.get("nb", 1),
            columns=instance_conf.get("na", 1),
            spacing=(instance_conf.get("dax", 0), instance_conf.get("dby", 0)),
        )

        instances[instance_name] = ref

    placements_conf = {} if placements_conf is None else placements_conf

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
            routing_strategy_name = routes_dict.pop("routing_strategy", "route_bundle")
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
                # handle bus connection syntax
                if ":" in port_src_string:
                    try:
                        src, src0, src1 = (
                            s.strip() for s in port_src_string.split(":")
                        )
                    except ValueError:
                        raise ValueError(
                            f"When including a colon (:) in a port name, bus syntax is expected (ref_name,port_prefix:istart:iend). Got '{port_src_string}'"
                        )
                    try:
                        dst, dst0, dst1 = (
                            s.strip() for s in port_dst_string.split(":")
                        )
                    except ValueError:
                        raise ValueError(
                            f"When including a colon (:) in a port name, bus syntax is expected (ref_name,port_prefix:istart:iend). Got '{port_dst_string}'"
                        )
                    try:
                        instance_src_name, port_src_name = (
                            s.strip() for s in src.split(",")
                        )
                    except ValueError:
                        raise ValueError(
                            f"Expected a single comma in port specification (ref_name,port_name). Got {src}"
                        )
                    try:
                        instance_dst_name, port_dst_name = (
                            s.strip() for s in dst.split(",")
                        )
                    except ValueError:
                        raise ValueError(
                            f"Expected a single comma in port specification (ref_name,port_name). Got {dst}"
                        )

                    try:
                        src0 = int(src0)
                    except ValueError:
                        raise ValueError(
                            f"Expected an integer index after first colon. Got {src0} ({port_src_string})"
                        )
                    try:
                        src1 = int(src1)
                    except ValueError:
                        raise ValueError(
                            f"Expected an integer index after second colon. Got {src1} ({port_src_string})"
                        )
                    try:
                        dst0 = int(dst0)
                    except ValueError:
                        raise ValueError(
                            f"Expected an integer index after first colon. Got {dst0} ({port_dst_string})"
                        )
                    try:
                        dst1 = int(dst1)
                    except ValueError:
                        raise ValueError(
                            f"Expected an integer index after second colon. Got {dst1} ({port_dst_string})"
                        )

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

                    instance_src_name, src_ia, src_ib = _parse_maybe_arrayed_instance(
                        instance_src_name
                    )
                    instance_dst_name, dst_ia, dst_ib = _parse_maybe_arrayed_instance(
                        instance_dst_name
                    )
                    instance_src = instances[instance_src_name]
                    instance_dst = instances[instance_dst_name]

                    for port_src_name in ports1names:
                        if port_src_name not in instance_src.ports:
                            instance_src_port_names = [
                                p.name for p in instance_src.ports
                            ]
                            raise ValueError(
                                f"{port_src_name!r} not in {instance_src_port_names}"
                                f"for {instance_src_name!r} "
                            )
                        if src_ia is None:
                            src_port = instance_src.ports[port_src_name]
                        else:
                            src_port = instance_src.ports[
                                (port_src_name, src_ia, src_ib)
                            ]

                        ports1.append(src_port)

                    for port_dst_name in ports2names:
                        if port_dst_name not in instance_dst.ports:
                            instance_dst_port_names = [
                                p.name for p in instance_dst.ports
                            ]
                            raise ValueError(
                                f"{port_dst_name!r} not in {instance_dst_port_names}"
                                f"for {instance_dst_name!r}"
                            )

                        if dst_ia is None:
                            dst_port = instance_dst.ports[port_dst_name]
                        else:
                            dst_port = instance_dst.ports[
                                (port_dst_name, dst_ia, dst_ib)
                            ]
                        ports2.append(dst_port)

                else:
                    instance_src_name, port_src_name = port_src_string.split(",")
                    instance_dst_name, port_dst_name = port_dst_string.split(",")

                    instance_src_name = instance_src_name.strip()
                    instance_dst_name = instance_dst_name.strip()
                    port_src_name = port_src_name.strip()
                    port_dst_name = port_dst_name.strip()

                    instance_src_name, src_ia, src_ib = _parse_maybe_arrayed_instance(
                        instance_src_name
                    )
                    instance_dst_name, dst_ia, dst_ib = _parse_maybe_arrayed_instance(
                        instance_dst_name
                    )

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

                    # if port_src_name not in instance_src.ports:
                    #     raise ValueError(
                    #         f"{port_src_name!r} not in {list(instance_src.ports.keys())} for"
                    #         f" {instance_src_name!r} "
                    #     )

                    # if port_dst_name not in instance_dst.ports:
                    #     raise ValueError(
                    #         f"{port_dst_name!r} not in {list(instance_dst.ports.keys())} for"
                    #         f" {instance_dst_name!r}"
                    #     )

                    if src_ia is None:
                        src_port = instance_src.ports[port_src_name]
                    else:
                        src_port = instance_src.ports[(port_src_name, src_ia, src_ib)]

                    if dst_ia is None:
                        dst_port = instance_dst.ports[port_dst_name]
                    else:
                        dst_port = instance_dst.ports[(port_dst_name, dst_ia, dst_ib)]
                    ports1.append(src_port)
                    ports2.append(dst_port)
                    route_name = f"{port_src_string}:{port_dst_string}"
                    route_names.append(route_name)

            routing_function = routing_strategy[routing_strategy_name]
            routes_list = routing_function(
                c,
                ports1=ports1,
                ports2=ports2,
                **settings,
            )
            for route_name, route in zip(route_names, routes_list):
                routes[route_name] = route

    if ports_conf:
        if not hasattr(ports_conf, "items"):
            raise ValueError(f"{ports_conf} needs to be a dict")
        for port_name, instance_comma_port in ports_conf.items():
            if "," in instance_comma_port:
                instance_name, instance_port_name = instance_comma_port.split(",")
                instance_name = instance_name.strip()
                instance_name, ia, ib = _parse_maybe_arrayed_instance(instance_name)
                instance_port_name = instance_port_name.strip()
                if instance_name not in instances:
                    raise ValueError(
                        f"{instance_name!r} not in {list(instances.keys())}"
                    )
                instance = instances[instance_name]

                port_names = [p.name for p in instance.ports]
                if instance_port_name not in port_names:
                    raise ValueError(
                        f"{instance_port_name!r} not in {port_names} for"
                        f" {instance_name!r} "
                    )
                if ia is None:
                    inst_port = instance.ports[instance_port_name]
                else:
                    inst_port = instance.ports[(instance_port_name, ia, ib)]
                c.add_port(port_name, port=inst_port)

    c.routes = routes
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
        routing_strategy: route_bundle


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
        routing_strategy: route_bundle
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
        routing_strategy: route_bundle
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
         doe: mmi1x2
         settings:
             length_mmi: [2, 100]
             width_mmi: [4, 10]
       pack:
         do_permutations: True
         spacing: 100

    mzi_sweep:
       component: pack_doe
       settings:
         doe: mzi
         settings:
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
        routing_strategy: route_bundle
        settings:
            separation: 20
            layer: [31, 0]
            width: 10

        links:
            mzi,e2: tr,e1

    electrical2:
        routing_strategy: route_bundle
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

sample_connections = """
name: sample_connections

instances:
    wgw:
      component: straight
      settings:
        length: 1
    wgn:
      component: straight
      settings:
        length: 0.5

connections:
    wgw,o1: wgn,o2
"""

sample_docstring = """
name: sample_docstring

instances:
    mmi_bot:
      component: mmi1x2
      settings:
        width_mmi: 5
        length_mmi: 11
    mmi_top:
      component: mmi1x2
      settings:
        width_mmi: 6
        length_mmi: 22

placements:
    mmi_top:
        port: o1
        x: 0
        y: 0
    mmi_bot:
        port: o1
        x: mmi_top,o2
        y: mmi_top,o2
        dx: 40
        dy: -40
routes:
    optical:
        links:
            mmi_top,o3: mmi_bot,o1
"""

yaml_anchor = """
name: yaml_anchor
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
        port: o3
        x: 0
        y: 0
    mmi_long:
        port: o1
        x: mmi_short,east
        y: mmi_short,north
        dx : 10
        dy: 10
"""

mirror_demo = """
name: mirror_demo
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5
placements:
    mmi_long:
        x: 0
        y: 0
        mirror: o1
        rotation: 0
"""


pad_array = """
name: pad_array
instances:
    pad_array:
      component: pad
      na: 3
      nb: 1
      dax: 200
      dby: 200

"""


if __name__ == "__main__":
    # c = from_yaml(sample_doe_function)
    # c = from_yaml(sample_mmis)
    c = from_yaml(sample_yaml_xmin)
    c.show()
    # n = c.get_netlist()
    # yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    # c2 = from_yaml(yaml_str)
    # n2 = c2.get_netlist()
    # c2.show()

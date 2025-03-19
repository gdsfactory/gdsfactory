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

import itertools
import pathlib
import re
import warnings
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import IO, TYPE_CHECKING, Any, Literal, Protocol, cast

import kfactory as kf
import networkx as nx
import yaml

from gdsfactory import typings
from gdsfactory.add_pins import add_instance_label
from gdsfactory.component import Component, ComponentReference
from gdsfactory.schematic import (
    Bundle,
    GridArray,
    Netlist,
    OrthogonalGridArray,
    Placement,
)
from gdsfactory.schematic import Instance as NetlistInstance
from gdsfactory.typings import LayerSpec, Route, RoutingStrategies

if TYPE_CHECKING:
    from gdsfactory.pdk import Pdk


class LabelInstanceFunction(Protocol):
    def __call__(
        self,
        component: Component,
        reference: ComponentReference,
        layer: LayerSpec | None = None,
        instance_name: str | None = None,
    ) -> None: ...


PlacementConf = dict[str, dict[str, int | float | str]]
ConnectionsByTransformedInst = dict[str, dict[str, str]]

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
    ref: ComponentReference, anchor_name: str
) -> tuple[float, float] | None:
    if anchor_name in valid_anchor_point_keywords:
        return cast(tuple[float, float], getattr(ref.dsize_info, anchor_name))
    elif anchor_name in ref.ports:
        return ref.ports[anchor_name].center
    return None


def _get_anchor_value_from_name(
    ref: ComponentReference, anchor_name: str, return_value: str
) -> float | None:
    """Return the x or y value of an anchor point or port on a reference."""
    if anchor_name in valid_anchor_value_keywords:
        return float(getattr(ref.dsize_info, anchor_name))
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
    placements_conf: PlacementConf,
    connections_by_transformed_inst: ConnectionsByTransformedInst,
    instances: dict[str, ComponentReference],
    encountered_insts: list[str],
    all_remaining_insts: list[str],
) -> float | None:
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


def _parse_maybe_arrayed_instance(inst_spec: str) -> tuple[str, int | None, int | None]:
    """Parse an instance specifier that may or may not be arrayed.

    Returns the instance name, and the a and b indices if they are present.
    """
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
            ia_int = int(ia)
        except ValueError as e:
            raise ValueError(
                f"When parsing array reference specifier '{inst_spec}', got a non-integer index '{ia}'"
            ) from e
        try:
            ib_int = int(ib)
        except ValueError as exc:
            raise ValueError(
                f"When parsing array reference specifier '{inst_spec}', got a non-integer index '{ib}'"
            ) from exc
        return inst_name, ia_int, ib_int
    return inst_spec, None, None


def place(
    placements_conf: dict[str, dict[str, int | float | str]],
    connections_by_transformed_inst: dict[str, dict[str, str]],
    instances: dict[str, ComponentReference],
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

        assert isinstance(rotation, int | float | None), "rotation must be a number"
        assert isinstance(port, str | None), "port must be a string or None"

        if rotation:
            if port:
                ref.rotate(rotation, center=_get_anchor_point_from_name(ref, port))
            else:
                ref.rotate(rotation)

        if mirror:
            if mirror is True and port:
                ref.dmirror_x(x=_get_anchor_value_from_name(ref, port, "x") or 0)
            elif mirror is True:
                ref.dcplx_trans *= kf.kdb.DCplxTrans(1, 0, True, 0, 0)
            elif mirror is False:
                pass
            elif isinstance(mirror, str):
                x_mirror = ref.ports[mirror].x
                ref.dmirror_x(x_mirror)
            elif isinstance(mirror, int | float):
                ref.dmirror_x(x=ref.x)
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
            ref.x -= a[0]
            ref.y -= a[1]

        if x is not None:
            _dx = _move_ref(
                x,
                x_or_y="x",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )
            assert _dx is not None
            ref.x += _dx

        if y is not None:
            _dy = _move_ref(
                y,
                x_or_y="y",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )
            assert _dy is not None
            ref.y += _dy

        if ymin is not None and ymax is not None:
            raise ValueError("You cannot set ymin and ymax")
        elif ymax is not None:
            dymax = _move_ref(
                ymax,
                x_or_y="y",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )
            assert dymax is not None
            ref.ymax = dymax
        elif ymin is not None:
            dymin = _move_ref(
                ymin,
                x_or_y="y",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )
            assert dymin is not None
            ref.ymin = dymin

        if xmin is not None and xmax is not None:
            raise ValueError("You cannot set xmin and xmax")
        elif xmin is not None:
            dxmin = _move_ref(
                xmin,
                x_or_y="x",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )
            assert dxmin is not None
            ref.xmin = dxmin
        elif xmax is not None:
            dxmax = _move_ref(
                xmax,
                x_or_y="x",
                placements_conf=placements_conf,
                connections_by_transformed_inst=connections_by_transformed_inst,
                instances=instances,
                encountered_insts=encountered_insts,
                all_remaining_insts=all_remaining_insts,
            )
            assert dxmax is not None
            ref.xmax = dxmax
        if dx:
            ref.x += float(dx)

        if dy:
            ref.y += float(dy)

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

        make_connection(instances=instances, **conn_info)  # type: ignore[arg-type]


def transform_connections_dict(
    connections_conf: dict[str, str],
) -> dict[str, dict[str, str | int | None]]:
    """Returns Dict with source_instance_name key and connection properties."""
    if not connections_conf:
        return {}
    attrs_by_src_inst: dict[str, dict[str, str | int | None]] = {}
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
    instances: dict[str, ComponentReference],
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

    if src_ia is None or src_ib is None:
        src_port = instance_src.ports[port_src_name]
    else:
        src_port = instance_src.ports[port_src_name, src_ia, src_ib]

    if dst_ia is None or dst_ib is None:
        dst_port = instance_dst.ports[port_dst_name]
    else:
        dst_port = instance_dst.ports[port_dst_name, dst_ia, dst_ib]

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
        settings:
            cross_section: strip

ports:
    o1: mmi_short,o1
    o2: mmi_long,o2
    o3: mmi_long,o3
"""


def cell_from_yaml(
    yaml_str: str | pathlib.Path | IO[Any] | dict[str, Any],
    routing_strategies: RoutingStrategies | None = None,
    label_instance_function: LabelInstanceFunction = add_instance_label,
    name: str | None = None,
) -> Callable[[], Component]:
    """Returns Component factory from YAML string or file.

    YAML includes instances, placements, routes, ports and connections.

    Args:
        yaml_str: YAML string or file.
        routing_strategies: for each route.
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
    routing_strategies = routing_strategies or {}

    return partial(
        from_yaml,
        yaml_str=yaml_str,
        routing_strategies=routing_strategies,
        label_instance_function=label_instance_function,
        name=name,
    )


def from_yaml(
    yaml_str: str | pathlib.Path | IO[Any] | dict[str, Any],
    routing_strategies: RoutingStrategies | None = None,
    label_instance_function: LabelInstanceFunction = add_instance_label,
    name: str | None = None,
) -> Component:
    """Returns Component from YAML string or file.

    YAML includes instances, placements, routes, ports and connections.

    Args:
        yaml_str: YAML string or file.
        routing_strategies: for each route.
        label_instance_function: to label each instance.
        name: Optional name.

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
    from gdsfactory.pdk import get_active_pdk

    routing_strategies = routing_strategies or {}

    c = Component()
    dct = _load_yaml_str(yaml_str)
    pdk = get_active_pdk()
    net = Netlist.model_validate(dct)
    g = _get_dependency_graph(net)
    refs = _get_references(c, pdk, net.instances)
    _place_and_connect(g, refs, net.connections, net.placements)
    c = _add_routes(c, refs, net.routes, routing_strategies)
    c = _add_ports(c, refs, net.ports)
    c = _add_labels(c, refs, label_instance_function)
    c.name = name or net.name or c.name
    return c


def _load_yaml_str(yaml_str: Any) -> dict[str, Any]:
    dct: dict[str, Any] = {}
    if isinstance(yaml_str, dict):
        dct = deepcopy(yaml_str)
    elif isinstance(yaml_str, Netlist):
        dct = deepcopy(yaml_str.model_dump())
    elif (isinstance(yaml_str, str) and "\n" in yaml_str) or isinstance(yaml_str, IO):
        dct = yaml.load(yaml_str, Loader=yaml.FullLoader)
    elif isinstance(yaml_str, str):
        dct = yaml.load(open(yaml_str), Loader=yaml.FullLoader)
    elif isinstance(yaml_str, pathlib.Path):
        dct = yaml.load(open(yaml_str), Loader=yaml.FullLoader)
    else:
        raise ValueError("Invalid format for 'yaml_str'.")
    return dct


def _get_dependency_graph(net: Netlist) -> nx.DiGraph:
    g = nx.DiGraph()

    for i, inst in net.instances.items():
        g.add_node(i)
        if isinstance(inst.array, OrthogonalGridArray):
            if inst.array.rows >= 2 or inst.array.columns >= 2:
                for a, b in itertools.product(
                    range(inst.array.rows), range(inst.array.columns)
                ):
                    _graph_connect(g, f"{i}<{a}.{b}>", i)
        elif isinstance(inst.array, GridArray):
            if inst.array.num_a >= 2 or inst.array.num_b >= 2:
                for a, b in itertools.product(
                    range(inst.array.num_a), range(inst.array.num_b)
                ):
                    _graph_connect(g, f"{i}<{a}.{b}>", i)

    for ip1, ip2 in net.connections.items():
        i1, _ = ip1.split(",")
        i2, _ = ip2.split(",")
        _graph_connect(g, i1, i2)

    for i1, pl in net.placements.items():
        for k, v in pl:
            if k not in ["x", "y", "xmin", "ymin", "xmax", "ymax"]:
                continue
            if not isinstance(v, str):
                continue
            if "," not in v:
                continue
            i2, _ = v.split(",")
            _graph_connect(g, i1, i2)

    cycles = list(nx.simple_cycles(g))
    if cycles:
        raise RuntimeError(
            "Cyclical references when placing / connecting instances:\n"
            + "\n".join("->".join(cyc + cyc[:1]) for cyc in cycles)
        )
    return g


def _get_references(
    c: Component, pdk: "Pdk", instances: dict[str, NetlistInstance]
) -> dict[str, ComponentReference]:
    refs: dict[str, ComponentReference] = {}
    for name, inst in instances.items():
        comp = pdk.get_component(component=inst.component, settings=inst.settings)
        if isinstance(inst.array, OrthogonalGridArray):
            ref = c.add_ref(
                comp,
                rows=inst.array.rows,
                columns=inst.array.columns,
                name=name,
                column_pitch=inst.array.column_pitch,
                row_pitch=inst.array.row_pitch,
            )
        elif isinstance(inst.array, GridArray):
            ref = c.create_inst(
                comp,
                na=inst.array.num_a,
                nb=inst.array.num_b,
                a=kf.kdb.DVector(
                    inst.array.pitch_a[0],
                    inst.array.pitch_a[1],
                ),
                b=kf.kdb.DVector(
                    inst.array.pitch_b[0],
                    inst.array.pitch_b[1],
                ),
            )
        else:
            ref = c.add_ref(comp, name=name)
        refs[name] = ref
    return refs


def _place_and_connect(
    g: nx.DiGraph,
    refs: dict[str, ComponentReference],
    connections: dict[str, str],
    placements: dict[str, Placement],
) -> None:
    directed_connections = _get_directed_connections(connections)

    for root in _graph_roots(g):
        pl = placements.get(root)
        if pl is not None:
            _update_reference_by_placement(refs, root, pl)
        for i2, i1 in nx.dfs_edges(g, root):
            ports = directed_connections.get(i1, {}).get(i2, None)
            pl = placements.get(i1)
            if pl is not None:
                _update_reference_by_placement(refs, i1, pl)
            if ports is not None:  # no elif!
                p1, p2 = ports
                i2name, i2a, i2b = _parse_maybe_arrayed_instance(i2)
                i1name, i1a, i1b = _parse_maybe_arrayed_instance(i1)

                for i in [i1name, i2name]:
                    if i not in refs:
                        raise ValueError(f"{i!r} not in {list(refs)}")

                if i1a is not None and i1b is not None:
                    port1 = refs[i1name].ports[p1, i1a, i1b]
                    if i2a is not None and i2b is not None:
                        refs[i1name].connect(port1, refs[i2name].ports[p2, i2a, i2b])
                    else:
                        if i2 not in refs:
                            raise ValueError(f"{i2!r} not in {list(refs)}")
                        refs[i1name].connect(port1, refs[i2].ports[p2])

                else:
                    if i2a is not None and i2b is not None:
                        if i1 not in refs:
                            raise ValueError(f"{i1!r} not in {list(refs)}")
                        refs[i1].connect(p1, refs[i2name].ports[p2, i2a, i2b])
                    else:
                        if i1 not in refs:
                            raise ValueError(f"{i1!r} not in {list(refs)}")
                        if i2 not in refs:
                            raise ValueError(f"{i2!r} not in {list(refs)}")
                        refs[i1].connect(p1, refs[i2].ports[p2])


def _add_routes(
    c: Component,
    refs: dict[str, ComponentReference],
    routes: dict[str, Bundle],
    routing_strategies: RoutingStrategies | None = None,
) -> Component:
    """Add routes to component."""
    from gdsfactory.pdk import get_routing_strategies

    routes_dict: dict[str, Route] = {}
    routing_strategies = routing_strategies or get_routing_strategies()
    for bundle_name, bundle in routes.items():
        try:
            routing_strategy = routing_strategies[bundle.routing_strategy]
        except KeyError as e:
            raise ValueError(
                f"Unknown routing strategy.\nvalid strategies: {list(routing_strategies)}\n"
                f"Got:{bundle.routing_strategy}"
            ) from e

        ports1: list[typings.Port] = []
        ports2: list[typings.Port] = []
        route_names: list[str] = []

        for ip1, ip2 in bundle.links.items():
            first1, middles1, last1 = _split_route_link(ip1)
            first2, middles2, last2 = _split_route_link(ip2)
            if len(middles1) != len(middles2):
                raise ValueError(
                    f"length of array bundles don't match. Got {ip1} <-> {ip2}"
                )
            ports1 += _get_ports_from_portnames(refs, first1, middles1, last1)
            ports2 += _get_ports_from_portnames(refs, first2, middles2, last2)
            route_names += [
                f"{bundle_name}-{first1}{m1}{last1}-{first2}{m2}{last2}"
                for m1, m2 in zip(middles1, middles2)
            ]
        routes_list = routing_strategy(
            c,
            ports1=ports1,
            ports2=ports2,
            **bundle.settings,
        )
        c.plot()
        routes_dict.update(dict(zip(route_names, routes_list)))
        c.routes = routes_dict
    return c


def _add_ports(
    c: Component, refs: dict[str, ComponentReference], ports: dict[str, str]
) -> Component:
    for name, ip in ports.items():
        i, p = (x.strip() for x in ip.split(","))
        i, ia, ib = _parse_maybe_arrayed_instance(i)
        if i not in refs:
            raise ValueError(f"{i!r} not in {list(refs)}")
        ref = refs[i]
        ps = [p.name for p in ref.ports]
        if p not in ps:
            raise ValueError(f"{p!r} not in {ps} for {i!r}.")
        inst_port = ref.ports[p] if ia is None else ref.ports[p, ia, ib]  # type: ignore[index]
        c.add_port(name, port=inst_port)
    return c


def _add_labels(
    c: Component,
    refs: dict[str, ComponentReference],
    label_instance_function: LabelInstanceFunction,
) -> Component:
    for name, ref in refs.items():
        label_instance_function(component=c, instance_name=name, reference=ref)
    return c


def _graph_roots(g: nx.DiGraph) -> list[str]:
    return [node for node in g.nodes if g.in_degree(node) == 0]


def _graph_connect(g: nx.DiGraph, i1: str, i2: str) -> None:
    g.add_edge(i2, i1)


def _two_out_of_three_none(one: Any, two: Any, three: Any) -> bool:
    if one is None:
        if two is None:
            return True
        if three is None:
            return True
    return two is None and three is None


def _update_reference_by_placement(
    refs: dict[str, ComponentReference], name: str, p: Placement
) -> None:
    ref = refs[name]
    x = p.x
    y = p.y
    xmin = p.xmin
    ymin = p.ymin
    xmax = p.xmax
    ymax = p.ymax
    dx = p.dx
    dy = p.dy
    port = p.port
    rotation = p.rotation
    mirror = p.mirror
    port_names = [port.name for port in ref.ports]

    if rotation:
        if isinstance(port, str):
            ref.rotate(rotation, center=_get_anchor_point_from_name(ref, port))
        else:
            ref.rotate(rotation)

    if mirror:
        if mirror is True:
            if isinstance(port, str):
                anchor_x = _get_anchor_value_from_name(ref, port, "x")
                assert anchor_x is not None, f"anchor_x is None for {port!r}"
                ref.dmirror_x(x=anchor_x)
            else:
                ref.dcplx_trans *= kf.kdb.DCplxTrans(1, 0, True, 0, 0)
        elif isinstance(mirror, str) and mirror in port_names:
            x_mirror = ref.ports[mirror].x
            ref.dmirror_x(x_mirror)
        else:
            try:
                mirror = float(mirror)
                ref.dmirror_x(x=ref.x)
            except Exception as e:
                raise ValueError(
                    f"{mirror!r} should be bool | float | str in {port_names}. Got: {mirror}."
                ) from e

    if isinstance(port, str):
        if xmin is not None or xmax is not None or ymin is not None or ymax is not None:
            raise ValueError(
                "Cannot combine 'port' setting with any of (xmin, xmax, ymin, ymax)."
                f"Got:\n{port=},\n{xmin=},\n{xmax=},\n{ymin=},\n{ymax=}"
            )
        a = _get_anchor_point_from_name(ref, port)
        if a is None:
            raise ValueError(
                f"Port {port!r} is neither a valid port on {ref.name!r} "
                "nor a recognized anchor keyword.\n"
                f"Valid ports: {port_names}. \n"
                f"Valid keywords: {valid_anchor_point_keywords}.\n"
                f"Got: {port}",
            )
        ref.x -= a[0]
        ref.y -= a[1]

    if not _two_out_of_three_none(x, xmin, xmax):
        raise ValueError(
            f"Can only set one of x, xmin, xmax. Got: {x=}, {xmin=}, {xmax=}"
        )
    elif isinstance(x, str):
        i, q = x.split(",")
        if q in valid_anchor_value_keywords:
            _dx = _get_anchor_value_from_name(refs[i], q, "x")
            assert _dx is not None, f"dx is None for {i!r}, {q!r}"
            ref.x += _dx
        else:
            ref.x += float(refs[i].ports[q].x)
    elif x is not None:
        ref.x += float(x)
    elif isinstance(xmin, str):
        i, q = xmin.split(",")
        if q in valid_anchor_value_keywords:
            dxmin = _get_anchor_value_from_name(refs[i], q, "x")
            assert dxmin is not None, f"dxmin is None for {i!r}, {q!r}"
            ref.xmin = dxmin
        else:
            ref.xmin = float(refs[i].ports[q].x)
    elif xmin is not None:
        ref.xmin = float(xmin)
    elif isinstance(xmax, str):
        i, q = xmax.split(",")
        if q in valid_anchor_value_keywords:
            dxmax = _get_anchor_value_from_name(refs[i], q, "x")
            assert dxmax is not None, f"dxmax is None for {i!r}, {q!r}"
            ref.xmax = dxmax
        else:
            ref.xmax = float(refs[i].ports[q].x)
    elif xmax is not None:
        ref.xmax = float(xmax)

    if not _two_out_of_three_none(y, ymin, ymax):
        raise ValueError(
            f"Can only set one of y, ymin, ymax. Got: {y=}, {ymin=}, {ymax=}"
        )
    elif isinstance(y, str):
        i, q = y.split(",")
        if q in valid_anchor_value_keywords:
            _dy = _get_anchor_value_from_name(refs[i], q, "y")
            assert _dy is not None, f"dy is None for {i!r}, {q!r}"
            ref.y += _dy
        else:
            ref.y += float(refs[i].ports[q].y)
    elif y is not None:
        ref.y += float(y)
    elif isinstance(ymin, str):
        i, q = ymin.split(",")
        if q in valid_anchor_value_keywords:
            dymin = _get_anchor_value_from_name(refs[i], q, "y")
            assert dymin is not None, f"dymin is None for {i!r}, {q!r}"
            ref.ymin = dymin
        else:
            ref.ymin = float(refs[i].ports[q].y)
    elif ymin is not None:
        ref.ymin = float(ymin)
    elif isinstance(ymax, str):
        i, q = ymax.split(",")
        if q in valid_anchor_value_keywords:
            dymax = _get_anchor_value_from_name(refs[i], q, "y")
            assert dymax is not None, f"dymax is None for {i!r}, {q!r}"
            ref.ymax = dymax
        else:
            ref.ymax = float(refs[i].ports[q].y)
    elif ymax is not None:
        ref.ymax = float(ymax)

    if dx is not None:
        ref.x += float(dx)

    if dy is not None:
        ref.y += float(dy)


def _get_directed_connections(
    connections: dict[str, str],
) -> dict[str, dict[str, tuple[str, str]]]:
    ret: dict[str, dict[str, tuple[str, str]]] = {}
    for ip1, ip2 in connections.items():
        i1, p1 = ip1.split(",")
        i2, p2 = ip2.split(",")
        if i1 not in ret:
            ret[i1] = {}
        ret[i1][i2] = (p1, p2)
    return ret


def _split_route_link(s: str) -> tuple[str, list[str], str]:
    error = ValueError(
        f"Invalid instance port format: {s!r}."
        "The format for a link instance port is 'inst,port',\n"
        "Whereas the format for bundle routing instance ports are one of the following:\n"
        "1. 'inst,port{i}-{j}' (enumerate port index)\n"
        "2. 'inst{i}-{j},port' (enumerate instance index)\n"
        "3. 'inst<{i}-{j}.{k}>,port (enumerate array instance index)"
    )
    warning = (
        "Bundle format 'inst,port:{i}:{j}' (with two columns) has been "
        "deprecated. Please use 'inst,port{i}-{j}' (with a single dash)"
    )

    def _try_int(i: str) -> int:
        try:
            return int(i)
        except ValueError as e:
            raise error from e

    def _first_index(ip: str) -> tuple[str, int]:
        p = re.sub("[0-9][0-9]*$", "", ip)
        idx = re.sub(f"^{p}", "", ip)
        return p, _try_int(idx)

    def _second_index(ip: str) -> tuple[str, int]:
        p = re.sub("^[0-9][0-9]*", "", ip)
        idx = re.sub(f"{p}$", "", ip)
        return p, _try_int(idx)

    if ":" in s:
        if s.count(":") == 2:
            s = s.replace(":", "", 1)
            s = s.replace(":", "-", 1)
            warnings.warn(warning)
        else:
            raise error

    if s.count(",") != 1:
        raise ValueError(f"Exactly one ',' expected in a route bundle link. Got: {s!r}")

    if s.count("-") > 1:
        raise error
    elif "-" not in s:
        return s, [""], ""
    else:
        first, last = s.split("-")
        first, j = _first_index(first)
        last, k = _second_index(last)

        if k >= j:
            middles = [f"{i}" for i in range(j, k + 1, 1)]
        else:
            middles = [f"{i}" for i in range(j, k - 1, -1)]
        return first, middles, last


def _get_ports_from_portnames(
    refs: dict[str, ComponentReference], first: str, middles: list[str], last: str
) -> list[typings.Port]:
    ports = []
    for middle in middles:
        ip = first + middle + last
        i, p = ip.split(",")
        i, ia, ib = _parse_maybe_arrayed_instance(i)
        ref = refs[i]
        if p not in ref.ports:
            raise ValueError(
                f"{p!r} not in {i!r} available ports: {[p.name for p in ref.ports]}"
            )
        port = ref.ports[p] if (ia is None or ib is None) else ref.ports[p, ia, ib]
        ports.append(port)
    return ports


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
          steps: [dx: 30, dy: '${settings.y}', dx: 20]
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
         do_permutations: True
         spacing: 100
         function: add_fiber_array

    mzi_sweep:
       component: pack_doe
       settings:
         doe: mzi
         settings:
            delta_length: [10, 100]
         do_permutations: True
         spacing: 100
         function: add_fiber_array

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
      columns: 3
      column_pitch: 200

"""
sample_array = """
name: sample_array

instances:
  sa1:
    component: straight
    columns: 5
    column_pitch: 50
    rows: 4
    row_pitch: 10
  s2:
    component: straight

connections:
    s2,o2: sa1<2.3>,o1

routes:
    b1:
        links:
            sa1<3.0>,o2: sa1<4.0>,o1
            sa1<3.1>,o2: sa1<4.1>,o1
        settings:
            cross_section: strip

ports:
    o1: s2,o1
    o2: sa1<0.0>,o1
"""
sample_mirror_simple = """
name: sample_mirror_simple

instances:
    s:
        component: straight

    b:
        component: bend_circular

placements:
    b:
        mirror: True
        port: o1

connections:
    b,o1: s,o2

"""

sample_doe = """
name: mask

instances:
    mmi1x2_sweep:
       component: pack_doe
       settings:
         doe: mmi1x2
         do_permutations: True
         spacing: 100
         settings:
           length_mmi: [2, 100]
           width_mmi: [4, 10]
"""

sample_2x2_connections = """
name: connections_2x2_solution

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
        links:
            mmi_bottom,o4: mmi_top,o1
            mmi_bottom,o3: mmi_top,o2

        settings:
            cross_section:
                cross_section: strip

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

same_placement = """
name: yaml_anchor
instances:
    mzi1:
      component: mzi
    mzi2:
      component: mzi
"""

port_array_electrical = """
instances:
  t:
    component: pad_array
    settings:
      port_orientation: 270
      columns: 10
      auto_rename_ports: True
  b:
    component: pad_array
    settings:
      port_orientation: 90
      columns: 10
      auto_rename_ports: True

placements:
  t:
    x: 500
    y: 900

routes:
  electrical:
    settings:
      start_straight_length: 150
      end_straight_length: 150
      cross_section: metal_routing
      allow_width_mismatch: True
      sort_ports: True
    links:
      t,e10-1: b,e1-10
"""

port_array_electrical2 = """
instances:
  t:
    component: pad
    settings:
      port_orientations:
        - 270
      port_orientation: null
      port_type: electrical
    array:
      columns: 3
      column_pitch: 150
  b:
    component: pad
    settings:
      port_orientations:
        - 90
      port_orientation: null
      port_type: electrical
    array:
      columns: 3
      column_pitch: 150

placements:
  t:
    x: 500
    y: 900

routes:
  electrical:
    settings:
      start_straight_length: 150
      end_straight_length: 150
      cross_section: metal_routing
      allow_width_mismatch: True
      sort_ports: True
    links:
      t<0-2.0>,e1: b<2-0.0>,e1
"""

port_array_optical = """
instances:
  a:
    component: nxn
  b:
    component: nxn

placements:
  b:
    x: 50
    y: 50
    rotation: 180
    # mirror: True
    mirror: False

routes:
  optical:
    settings:
        cross_section: strip
    links:
      a,o3-4: b,o4-3
"""

mirror = """
instances:
  a:
    component: bend_circular

placements:
  a:
    # rotation: 180
    mirror: True
    # mirror: False
"""

sample_array_connect_error = """
name: sample_array_connect_error

instances:
  b1:
    component: bend_euler
    settings:
      radius: 20
  s1:
    component: straight
    settings:
      length: 10
    array:
      columns: 3
      rows: 1
      column_pitch: 100.0
      row_pitch: 0.0
connections:
  #s1<2.0>,o2: b1,o1
  b1,o1: s2<2.0>,o2
"""

if __name__ == "__main__":
    # c = from_yaml(sample_array)
    # c = from_yaml(sample_array_connect_error)
    # c = from_yaml(sample_yaml_xmin)
    c = from_yaml(sample_doe_function)
    # n = c.get_netlist()
    c.show()
    # yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    # c2 = from_yaml(yaml_str)
    # n2 = c2.get_netlist()
    # c2.show()

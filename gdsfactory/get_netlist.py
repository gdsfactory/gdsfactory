"""Extract netlist from component port connectivity.

Assumes two ports are connected when they have same width, x, y

.. code:: yaml

    connections:
        - coupler,N0:bendLeft,W0
        - coupler,N1:bendRight,N0
        - bednLeft,N0:straight,W0
        - bendRight,N0:straight,E0

    ports:
        - coupler,E0
        - coupler,W0

"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from pprint import pprint
from typing import Any, Protocol
from warnings import warn

import numpy as np
from kfactory import DKCell, LayerEnum

from gdsfactory import Port, typings
from gdsfactory.component import (
    Component,
    ComponentReference,
    ComponentReferences,
)
from gdsfactory.name import clean_name
from gdsfactory.serialization import clean_dict, clean_value_json
from gdsfactory.typings import LayerSpec


def nets_to_connections(
    nets: list[dict[str, Any]], connections: dict[str, Any]
) -> dict[str, str]:
    connections = dict(connections)
    inverse_connections = {v: k for k, v in connections.items()}

    def _is_connected(p: str) -> bool:
        return (p in connections) or (p in inverse_connections)

    def _add_connection(p: str, q: str) -> None:
        connections[p] = q
        inverse_connections[q] = p

    def _get_connected_port(p: str) -> Any:
        return connections[p] if p in connections else inverse_connections[p]

    for net in nets:
        p = net["p1"]
        q = net["p2"]
        if _is_connected(p):
            _q = _get_connected_port(p)
            raise ValueError(
                "SAX currently does not support multiply connected ports. "
                f"Got {p}<->{q} and {p}<->{_q}"
            )
        if _is_connected(q):
            _p = _get_connected_port(q)
            raise ValueError(
                "SAX currently does not support multiply connected ports. "
                f"Got {p}<->{q} and {_p}<->{q}"
            )
        _add_connection(p, q)
    return connections


def get_default_connection_validators() -> dict[str, Callable[..., None]]:
    return {"optical": validate_optical_connection, "electrical": _null_validator}


def get_instance_name_from_alias(reference: ComponentReference) -> str:
    """Returns the instance name from the label.

    If no label returns to instanceName_x_y.

    Args:
        reference: reference that needs naming.
    """
    return clean_name(reference.name or "")


def get_instance_name_from_label(
    component: Component,
    reference: ComponentReference,
    layer_label: LayerSpec = "LABEL_INSTANCE",
) -> str:
    """Returns the instance name from the label.

    If no label returns to instanceName_x_y.

    Args:
        component: with labels.
        reference: reference that needs naming.
        layer_label: ignores layer_label[1].
    """
    from gdsfactory.pdk import get_layer

    layer_label = get_layer(layer_label)
    layer = layer_label[0] if isinstance(layer_label, LayerEnum) else layer_label

    x = reference.dx
    y = reference.dy
    labels = component.labels

    # default instance name follows component.aliases
    text = clean_name(f"{reference.cell.name}_{x}_{y}")

    # try to get the instance name from a label
    for label in labels:
        xl = label.dposition[0]
        yl = label.dposition[1]
        if x == xl and y == yl and label.layer == layer:
            # print(label.text, xl, yl, x, y)
            return str(label.text)

    return text


def _is_array_reference(ref: ComponentReference) -> bool:
    return ref.na > 1 or ref.nb > 1


def _is_orthogonal_array_reference(ref: ComponentReference) -> bool:
    return abs(ref.a.y) == 0 and abs(ref.b.x) == 0


def get_netlist(
    component: DKCell,
    exclude_port_types: Sequence[str] | None = (
        "placement",
        "pad",
        "bump",
        "vertical_te",
        "vertical_tm",
        "edge_coupler",
    ),
    get_instance_name: Callable[..., str] = get_instance_name_from_alias,
    allow_multiple: bool = True,
    connection_error_types: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """From Component returns a dict with instances, connections and placements.

    warnings collected during netlisting are reported back into the netlist.
    These include warnings about mismatched port widths, orientations, shear angles, excessive offsets, etc.
    You can also configure warning types which should throw an error when encountered
    by modifying connection_error_types.
    A key difference in this algorithm is that we group each port type independently.
    This allows us to use different logic to determine i.e.
    if an electrical port is properly connected vs an optical port.
    In this function, the core logic is the same, but we employ extra validation for optical ports.

    Args:
        component: to extract netlist.
        exclude_port_types: optional list of port types to exclude from netlisting.
        get_instance_name: function to get instance name.
        allow_multiple: False to raise an error if more than two ports share the same connection. \
                if True, will return key: [value] pairs with [value] a list of all connected instances.
        connection_error_types: optional dictionary of port types and error types to raise an error for.

    Returns:
        instances: Dict of instance name and settings.
        nets: List of connected port pairs/groups
        placements: Dict of instance names and placements (x, y, rotation).
        port: Dict portName: ComponentName,port.
        name: name of component.
        warnings: warning messages (disconnected pins).

    """
    placements: dict[str, dict[str, Any]] = {}
    instances: dict[str, dict[str, Any]] = {}
    nets: list[dict[str, Any]] = []
    top_ports: dict[str, str] = {}

    # store where ports are located
    name2port: dict[str, typings.Port] = {}

    # TOP level ports
    ports = component.ports
    ports_by_type: defaultdict[str, list[str]] = defaultdict(list)
    top_ports_list: set[str] = set()

    references = _get_references_to_netlist(component)

    for reference in references:
        c = reference.cell
        origin = reference.dtrans.disp
        x = origin.x
        y = origin.y
        reference_name = get_instance_name(reference)
        instance: dict[str, Any] = {}

        if c.info:
            instance.update(component=c.name, info=c.info.model_dump())

        # Don't extract netlist for cells with no function_name (e.g. subcells imported from GDS)
        if not c.function_name:
            continue

        # Prefer name from settings over c.name
        if c.settings:
            settings = c.settings.model_dump()

            instance.update(
                component=c.function_name,
                settings=settings,
            )

        instances[reference_name] = instance
        placements[reference_name] = {
            "x": x,
            "y": y,
            "rotation": reference.dcplx_trans.angle,
            "mirror": reference.dtrans.mirror,
        }

        if _is_array_reference(reference):
            if _is_orthogonal_array_reference(reference):
                instances[reference_name]["array"] = {
                    "columns": reference.na,
                    "rows": reference.nb,
                    "column_pitch": reference.instance.da.x,
                    "row_pitch": reference.instance.db.y,
                }
            else:
                instances[reference_name]["array"] = {
                    "num_a": reference.na,
                    "num_b": reference.nb,
                    "pitch_a": (reference.instance.da.x, reference.instance.da.y),
                    "pitch_b": (reference.instance.db.x, reference.instance.db.y),
                }
            reference_name = get_instance_name(reference)
            for ia in range(reference.na):
                for ib in range(reference.nb):
                    for port in reference.cell.ports:
                        ref_port = reference.ports[port.name, ia, ib]
                        src = f"{reference_name}<{ia}.{ib}>,{port.name}"
                        name2port[src] = ref_port
                        ports_by_type[port.port_type].append(src)
        else:
            # lower level ports
            for port_ in reference.ports:
                reference_name = get_instance_name(reference)
                src = f"{reference_name},{port_.name}"
                name2port[src] = port_
                ports_by_type[port_.port_type].append(src)

    for port in ports:
        port_name = port.name
        if port_name is not None:
            name2port[port_name] = port
            top_ports_list.add(port_name)
            ports_by_type[port.port_type].append(port_name)

    warnings: dict[str, Any] = {}
    for port_type, port_names in ports_by_type.items():
        if exclude_port_types and port_type in exclude_port_types:
            continue
        connections_t, warnings_t = extract_connections(
            port_names,
            name2port,
            port_type,
            allow_multiple=allow_multiple,
            connection_error_types=connection_error_types,
        )
        if warnings_t:
            warnings[port_type] = warnings_t
        for connection in connections_t:
            if len(connection) == 2:
                src, dst = connection
                if src in top_ports_list:
                    top_ports[src] = dst
                elif dst in top_ports_list:
                    top_ports[dst] = src
                else:
                    src_dest = sorted([src, dst])
                    net = {"p1": src_dest[0], "p2": src_dest[1]}
                    nets.append(net)

    # sort nets by p1 (and then p2, in the case of a tie)
    nets_sorted = sorted(nets, key=lambda net: f"{net['p1']},{net['p2']}")
    placements_sorted = {k: placements[k] for k in sorted(placements.keys())}
    instances_sorted = {k: instances[k] for k in sorted(instances.keys())}
    netlist: dict[str, Any] = {
        "nets": nets_sorted,
        "instances": instances_sorted,
        "placements": placements_sorted,
        "ports": top_ports,
        "name": component.name,
    }
    if warnings:
        netlist["warnings"] = warnings
    return clean_value_json(netlist)  # type: ignore[no-any-return]


def extract_connections(
    port_names: Sequence[str],
    ports: dict[str, typings.Port],
    port_type: str,
    validators: dict[str, Callable[..., None]] | None = None,
    allow_multiple: bool = True,
    connection_error_types: dict[str, list[str]] | None = None,
) -> tuple[list[list[str]], dict[str, list[dict[str, Any]]]]:
    if validators is None:
        validators = DEFAULT_CONNECTION_VALIDATORS

    validator = validators.get(port_type, _null_validator)
    return _extract_connections(
        port_names,
        ports,
        port_type,
        connection_validator=validator,
        allow_multiple=allow_multiple,
        connection_error_types=connection_error_types,
    )


def _extract_connections(
    port_names: Sequence[str],
    ports: dict[str, typings.Port],
    port_type: str,
    connection_validator: Callable[..., None],
    raise_error_for_warnings: list[str] | None = None,
    allow_multiple: bool = True,
    connection_error_types: dict[str, list[str]] | None = None,
) -> tuple[list[list[str]], dict[str, list[Any]]]:
    """Extracts connections between ports.

    Args:
        port_names: list of port names.
        ports: dict of port names to Port objects.
        port_type: type of port.
        connection_validator: function to validate connections.
        raise_error_for_warnings: list of warning types to raise an error for.
        allow_multiple: False to raise an error if more than two ports share the same connection.
        connection_error_types: optional dictionary of port types and error types to raise an error for.

    """
    if connection_error_types is None:
        connection_error_types = DEFAULT_CRITICAL_CONNECTION_ERROR_TYPES

    warnings: defaultdict[str, list[Any]] = defaultdict(list)
    if raise_error_for_warnings is None:
        raise_error_for_warnings = connection_error_types.get(port_type, [])

    unconnected_port_names: list[str] = list(port_names)
    connections: list[list[str]] = []

    by_xy: dict[tuple[float, float], list[str]] = defaultdict(list)

    for port_name in unconnected_port_names:
        port = ports[port_name]
        by_xy[port.to_itype().center].append(port_name)

    unconnected_port_names = []

    for xy, ports_at_xy in by_xy.items():
        if len(ports_at_xy) == 1:
            unconnected_port_names.append(ports_at_xy[0])

        elif len(ports_at_xy) == 2:
            port1 = ports[ports_at_xy[0]]
            port2 = ports[ports_at_xy[1]]
            connection_validator(port1, port2, ports_at_xy, warnings)
            connections.append(ports_at_xy)

        elif not allow_multiple:
            warnings["multiple_connections"].append(ports_at_xy)
            warn(f"Found multiple connections at {xy}:{ports_at_xy}")

        else:
            # Iterates over the list of multiple ports to create related two-port connectivity
            num_ports = len(ports_at_xy)
            for portindex1, portindex2 in zip(
                range(-1, num_ports - 1), range(num_ports)
            ):
                port1 = ports[ports_at_xy[portindex1]]
                port2 = ports[ports_at_xy[portindex2]]
                connection_validator(port1, port2, ports_at_xy, warnings)
                connections.append([ports_at_xy[portindex1], ports_at_xy[portindex2]])

    if unconnected_port_names:
        unconnected_non_top_level = [
            pname for pname in unconnected_port_names if ("," in pname)
        ]
        if unconnected_non_top_level:
            unconnected_xys = [
                ports[pname].center for pname in unconnected_non_top_level
            ]
            warnings["unconnected_ports"].append(
                _make_warning(
                    ports=unconnected_non_top_level,
                    values=unconnected_xys,
                    message=f"{len(unconnected_non_top_level)} unconnected {port_type} ports!",
                )
            )

    critical_warnings = {
        w: warnings[w] for w in raise_error_for_warnings if w in warnings
    }

    if critical_warnings and raise_error_for_warnings:
        pprint(critical_warnings)
        warn("Found critical warnings while extracting netlist")
    return connections, dict(warnings)


def _make_warning(ports: list[str], values: Any, message: str) -> dict[str, Any]:
    w = {
        "ports": ports,
        "values": values,
        "message": message,
    }
    return clean_dict(w)


def _null_validator(
    port1: Port,
    port2: Port,
    port_names: list[str],
    warnings: dict[str, list[dict[str, Any]]],
) -> None:
    pass


def validate_optical_connection(
    port1: Port,
    port2: Port,
    port_names: list[str],
    warnings: dict[str, list[dict[str, Any]]],
    angle_tolerance: float = 0.01,
    offset_tolerance: float = 0.001,
    width_tolerance: float = 0.001,
) -> None:
    is_top_level = [("," not in pname) for pname in port_names]

    if len(port_names) != 2:
        raise ValueError(f"More than two connected optical ports: {port_names}")

    if all(is_top_level):
        raise ValueError(f"Two top-level ports appear to be connected: {port_names}")

    if abs(port1.width - port2.width) > width_tolerance:
        warnings["width_mismatch"].append(
            _make_warning(
                port_names,
                values=[port1.width, port2.width],
                message=f"Widths of ports {port_names[0]} and {port_names[1]} not equal. "
                f"Difference of {abs(port1.width - port2.width)} um",
            )
        )

    if any(is_top_level):
        if (
            abs(difference_between_angles(port1.orientation, port2.orientation))
            > angle_tolerance
        ):
            top_port, lower_port = port_names if is_top_level[0] else port_names[::-1]
            warnings["orientation_mismatch"].append(
                _make_warning(
                    port_names,
                    values=[port1.orientation, port2.orientation],
                    message=f"{lower_port} was promoted to {top_port} but orientations"
                    f"do not match! Difference of {(abs(port1.orientation - port2.orientation))} deg",
                )
            )
    else:
        angle_misalignment = abs(
            abs(difference_between_angles(port1.orientation, port2.orientation)) - 180
        )
        if angle_misalignment > angle_tolerance:
            warnings["orientation_mismatch"].append(
                _make_warning(
                    port_names,
                    values=[port1.orientation, port2.orientation],
                    message=f"{port_names[0]} and {port_names[1]} are misaligned by {angle_misalignment} deg",
                )
            )

    offset_mismatch = np.sqrt(np.sum(np.square(np.array(port2.center) - port1.center)))
    if offset_mismatch > offset_tolerance:
        warnings["offset_mismatch"].append(
            _make_warning(
                port_names,
                values=[port1.center, port2.center],
                message=f"{port_names[0]} and {port_names[1]} are offset by {offset_mismatch} um",
            )
        )


def difference_between_angles(angle2: float, angle1: float) -> float:
    diff = angle2 - angle1
    while diff < 180:
        diff += 360
    while diff > 180:
        diff -= 360
    return diff


def _get_references_to_netlist(component: DKCell) -> ComponentReferences:
    return component.insts


class GetNetlistFunc(Protocol):
    def __call__(self, component: DKCell, **kwargs: Any) -> dict[str, Any]: ...


def get_netlist_recursive(
    component: DKCell,
    component_suffix: str = "",
    get_netlist_func: GetNetlistFunc = get_netlist,  # type: ignore[assignment]
    get_instance_name: Callable[..., str] = get_instance_name_from_alias,
    **kwargs: Any,
) -> dict[str, Any]:
    """Returns recursive netlist for a component and subcomponents.

    Args:
        component: to extract netlist.
        component_suffix: suffix to append to each component name.
            useful if to save and reload a back-annotated netlist.
        get_netlist_func: function to extract individual netlists.
        get_instance_name: function to get instance name.
        kwargs: additional keyword arguments to pass to get_netlist_func.

    Keyword Args:
        tolerance: tolerance in grid_factor to consider two ports connected.
        exclude_port_types: optional list of port types to exclude from netlisting.
        get_instance_name: function to get instance name.

    Returns:
        Dictionary of netlists, keyed by the name of each component.

    """
    all_netlists: dict[str, Any] = {}

    # only components with references (subcomponents) warrant a netlist
    references = _get_references_to_netlist(component)

    if references:
        netlist = get_netlist_func(component, **kwargs)
        all_netlists[f"{component.name}{component_suffix}"] = netlist

        # for each reference, expand the netlist
        for ref in references:
            rcell = ref.cell
            grandchildren = get_netlist_recursive(
                component=rcell,
                component_suffix=component_suffix,
                get_netlist_func=get_netlist_func,
                **kwargs,
            )
            all_netlists |= grandchildren

            child_references = _get_references_to_netlist(ref.cell)

            if child_references:
                inst_name = get_instance_name(ref)
                netlist_dict: dict[str, Any] = {
                    "component": f"{rcell.name}{component_suffix}"
                }
                if hasattr(rcell, "settings"):
                    netlist_dict.update(settings=rcell.settings.model_dump())
                if hasattr(rcell, "info"):
                    netlist_dict.update(info=rcell.info.model_dump())
                netlist["instances"][inst_name] = netlist_dict

    return all_netlists


def _demo_ring_single_array() -> None:
    from gdsfactory.components.rings.ring_single_array import ring_single_array

    c = ring_single_array()
    c.get_netlist()


def _demo_mzi_lattice() -> None:
    import gdsfactory as gf

    coupler_lengths = [10, 20, 30, 40]
    coupler_gaps = [0.1, 0.2, 0.4, 0.5]
    delta_lengths = [10, 100, 200]

    c = gf.components.mzi_lattice(
        coupler_lengths=coupler_lengths,
        coupler_gaps=coupler_gaps,
        delta_lengths=delta_lengths,
    )
    c.get_netlist()


DEFAULT_CONNECTION_VALIDATORS = get_default_connection_validators()

DEFAULT_CRITICAL_CONNECTION_ERROR_TYPES = {
    "optical": ["width_mismatch", "shear_angle_mismatch", "orientation_mismatch"]
}


if __name__ == "__main__":
    from pprint import pprint

    import gdsfactory as gf

    # c = gf.Component()
    # mzi = c << gf.c.mzi()
    # mzi.dxmin = 10
    # mzi.name = "mzi"
    # bend = c << gf.c.bend_euler()
    # bend.connect("o1", mzi.ports["o2"])
    # bend.name = "bend"
    # c.add_port("o1", port=mzi.ports["o1"])
    # c.add_port("o2", port=bend.ports["o2"])

    c = gf.c.pad_array()
    # c = gf.components.array(
    #     gf.components.straight(length=100), spacing=(100, 0), columns=5, rows=1
    # )
    c.show()
    n0 = c.get_netlist()
    pprint(n0)

    # gdspath = c.write_gds("test.gds")
    # c = gf.import_gds(gdspath)
    # n = c.get_netlist()
    # pprint(n["placements"])
    c.show()

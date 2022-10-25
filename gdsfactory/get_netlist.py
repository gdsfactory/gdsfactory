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

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gdspy
import numpy as np
import omegaconf

from gdsfactory import Port
from gdsfactory.component import Component, ComponentReference
from gdsfactory.name import clean_name
from gdsfactory.pdk import get_layer
from gdsfactory.serialization import clean_dict, clean_value_json
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import LayerSpec


def get_default_connection_validators():
    return {"optical": validate_optical_connection, "electrical": _null_validator}


def get_instance_name_from_alias(
    component: Component,
    reference: ComponentReference,
) -> str:
    """Returns the instance name from the label.

    If no label returns to instanceName_x_y

    Args:
        component: with labels.
        reference: reference that needs naming.
    """
    return reference.name


def get_instance_name_from_label(
    component: Component,
    reference: ComponentReference,
    layer_label: LayerSpec = "LABEL_INSTANCE",
) -> str:
    """Returns the instance name from the label.

    If no label returns to instanceName_x_y

    Args:
        component: with labels.
        reference: reference that needs naming.
        layer_label: ignores layer_label[1].
    """
    layer_label = get_layer(layer_label)

    x = snap_to_grid(reference.x)
    y = snap_to_grid(reference.y)
    labels = component.labels

    # default instance name follows component.aliases
    text = clean_name(f"{reference.parent.name}_{x}_{y}")

    # try to get the instance name from a label
    for label in labels:
        xl = snap_to_grid(label.position[0])
        yl = snap_to_grid(label.position[1])
        if x == xl and y == yl and label.layer == layer_label[0]:
            # print(label.text, xl, yl, x, y)
            return label.text

    return text


def get_netlist_yaml(
    component: Component,
    full_settings: bool = False,
    tolerance: int = 5,
    exclude_port_types: Optional[List] = None,
    **kwargs,
) -> Dict:
    """Returns instances, connections and placements yaml string content."""
    return omegaconf.OmegaConf.to_yaml(
        get_netlist(
            component=component,
            full_settings=full_settings,
            tolerance=tolerance,
            exclude_port_types=exclude_port_types,
            **kwargs,
        )
    )


def get_netlist(
    component: Component,
    full_settings: bool = False,
    tolerance: int = 5,
    exclude_port_types: Optional[Union[List[str], Tuple[str]]] = ("placement",),
    get_instance_name: Callable[..., str] = get_instance_name_from_alias,
) -> Dict[str, Any]:
    """From Component returns instances, connections and placements dict.

    Does two sweeps over the connections:

    1. first tries to connect everything assuming perfect connections at each port.
    2. Then gathers ports which did not perfectly connect to anything and tries \
            to find imperfect connections, by grouping ports on a coarse grid.

    warnings collected during netlisting are reported back into the netlist.
    These include warnings about mismatched port widths, orientations, shear angles, excessive offsets, etc.
    You can also configure warning types which should throw an error when encountered by modifying DEFAULT_CRITICAL_CONNECTION_ERROR_TYPES.
    Validators, which will produce warnings for each port type, can be overridden with DEFAULT_CONNECTION_VALIDATORS
    A key difference in this algorithm is that we group each port type independently.
    This allows us to use different logic to determine i.e. if an electrical port is properly connected vs an optical port.
    In this function, the core logic is the same, but we employ extra validation for optical ports.
    snap_to_grid() allows a value of 0, which will return the original value,
    is more efficient when the value is 1, and will throw a more descriptive error when the value is <0
    the default value of tolerance is 5nm because it should allow better performance with the two-grid-sweep approach.


    Args:
        component: to extract netlist.
        full_settings: True returns all, false changed settings.
        tolerance: tolerance in nm to consider two ports connected.
        exclude_port_types: optional list of port types to exclude from netlisting.
        get_instance_name: function to get instance name.

    Returns:
        instances: Dict of instance name and settings.
        connections: Dict of Instance1Name,portName: Instace2Name,portName.
        placements: Dict of instance names and placements (x, y, rotation).
        port: Dict portName: ComponentName,port.
        name: name of component.
        warnings: warning messages (disconnected pins).

    """
    placements = {}
    instances = {}
    connections = {}
    top_ports = {}

    # store where ports are located
    name2port = {}

    # TOP level ports
    ports = component.get_ports(depth=0)
    ports_by_type = defaultdict(list)
    top_ports_list = set()

    references = _get_references_to_netlist(component)

    for reference in references:
        c = reference.parent
        origin = reference.origin
        x = float(snap_to_grid(origin[0]))
        y = float(snap_to_grid(origin[1]))
        reference_name = get_instance_name(
            component,
            reference,
        )
        if isinstance(reference, gdspy.CellArray):
            is_array = True
            base_reference_name = reference_name
            reference_name += "__1_1"
        else:
            is_array = False

        instance = {}

        if c.info:
            instance.update(component=c.name, info=clean_value_json(c.info))

        # Prefer name from settings over c.name
        if c.settings:
            settings = c.settings.full if full_settings else c.settings.changed

            instance.update(
                component=getattr(c.settings, "function_name", c.name),
                settings=clean_value_json(settings),
            )

        instances[reference_name] = instance
        placements[reference_name] = dict(
            x=x,
            y=y,
            rotation=int(reference.rotation or 0),
            mirror=reference.x_reflection or 0,
        )
        if is_array:
            parent_ports = c.ports
            for i in range(reference.rows):
                for j in range(reference.columns):
                    reference_name = f"{base_reference_name}__{i + 1}_{j + 1}"
                    xj = x + j * reference.spacing[0]
                    yi = y + i * reference.spacing[1]
                    instances[reference_name] = instance
                    placements[reference_name] = dict(
                        x=xj,
                        y=yi,
                        rotation=int(reference.rotation or 0),
                        mirror=reference.x_reflection or 0,
                    )
                    for parent_port_name in parent_ports:
                        top_name = f"{parent_port_name}_{i + 1}_{j + 1}"
                        lower_name = f"{reference_name},{parent_port_name}"
                        # a bit of a hack... get the top-level port for the ComponentArray, by our known naming convention. I hope no one renames these ports!
                        parent_port = component.ports[top_name]
                        name2port[lower_name] = parent_port
                        top_ports_list.add(top_name)
                        ports_by_type[parent_port.port_type].append(lower_name)

    for port in ports:
        src = port.name
        name2port[src] = port
        top_ports_list.add(src)
        ports_by_type[port.port_type].append(src)

    # lower level ports
    for reference in references:
        if not isinstance(reference, gdspy.CellArray):
            for port in reference.ports.values():
                reference_name = get_instance_name(
                    component,
                    reference,
                )
                src = f"{reference_name},{port.name}"
                name2port[src] = port
                ports_by_type[port.port_type].append(src)

    warnings = {}
    for port_type, port_names in ports_by_type.items():
        if exclude_port_types and port_type in exclude_port_types:
            continue
        connections_t, warnings_t = extract_connections(
            port_names, name2port, port_type, tolerance=tolerance
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
                    connections[src_dest[0]] = src_dest[1]

    connections_sorted = {k: connections[k] for k in sorted(list(connections.keys()))}
    placements_sorted = {k: placements[k] for k in sorted(list(placements.keys()))}
    instances_sorted = {k: instances[k] for k in sorted(list(instances.keys()))}
    netlist = dict(
        connections=connections_sorted,
        instances=instances_sorted,
        placements=placements_sorted,
        ports=top_ports,
        name=component.name,
    )
    if warnings:
        netlist["warnings"] = warnings
    return netlist


def extract_connections(
    port_names: List[str],
    ports: Dict[str, Port],
    port_type: str,
    tolerance: int = 5,
    validators: Optional[Dict[str, Callable]] = None,
):
    if validators is None:
        validators = DEFAULT_CONNECTION_VALIDATORS

    validator = validators.get(port_type, _null_validator)
    return _extract_connections_two_sweep(
        port_names,
        ports,
        port_type,
        tolerance=tolerance,
        connection_validator=validator,
    )


def _extract_connections_two_sweep(
    port_names: List[str],
    ports: Dict[str, Port],
    port_type: str,
    connection_validator: Callable,
    tolerance: int,
    raise_error_for_warnings: Optional[List[str]] = None,
):
    warnings = defaultdict(list)
    if raise_error_for_warnings is None:
        raise_error_for_warnings = DEFAULT_CRITICAL_CONNECTION_ERROR_TYPES.get(
            port_type, []
        )

    unconnected_port_names = list(port_names)
    if tolerance < 0:
        raise ValueError(f"Cannot have a tolerance less than zero. Got {tolerance}")
    elif tolerance <= 1:
        # if tolerance is 0 or 1, do only one sweep with that tolerance
        grids = [("fine", tolerance)]
    else:
        # default: do one fine sweep with a 1nm tolerance, then a coarse sweep with the given tolerance to connect any remaining ports which are not perfectly aligned
        grids = [("fine", 1), ("coarse", tolerance)]

    connections = []

    for _grid_name, grid_size in grids:
        by_xy = defaultdict(list)

        for port_name in unconnected_port_names:
            port = ports[port_name]
            by_xy[tuple(snap_to_grid(port.center, nm=grid_size))].append(port_name)

        unconnected_port_names = []

        for xy, ports_at_xy in by_xy.items():
            if len(ports_at_xy) == 1:
                unconnected_port_names.append(ports_at_xy[0])

            elif len(ports_at_xy) == 2:
                port1 = ports[ports_at_xy[0]]
                port2 = ports[ports_at_xy[1]]
                connection_validator(port1, port2, ports_at_xy, warnings)
                connections.append(ports_at_xy)

            else:
                warnings["multiple_connections"].append(ports_at_xy)
                raise ValueError(f"Found multiple connections at {xy}:{ports_at_xy}")

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

    if critical_warnings:
        raise ValueError(
            f"Found critical warnings while extracting netlist: {critical_warnings}"
        )
    return connections, dict(warnings)


def _make_warning(ports: List[str], values: Any, message: str) -> Dict[str, Any]:
    w = {
        "ports": ports,
        "values": values,
        "message": message,
    }
    return clean_dict(w)


def _null_validator(port1: Port, port2: Port, port_names, warnings):
    pass


def validate_optical_connection(
    port1: Port,
    port2: Port,
    port_names,
    warnings,
    angle_tolerance=0.01,
    offset_tolerance=0.001,
    width_tolerance=0.001,
):
    is_top_level = [("," not in pname) for pname in port_names]

    if all(is_top_level):
        raise ValueError(f"Two top-level ports appear to be connected: {port_names}")

    if abs(port1.width - port2.width) > width_tolerance:
        warnings["width_mismatch"].append(
            _make_warning(
                port_names,
                values=[port1.width, port2.width],
                message=f"Widths of ports {port_names[0]} and {port_names[1]} not equal. Difference of {abs(port1.width - port2.width)} um",
            )
        )
    if port1.shear_angle and not port2.shear_angle:
        warnings["shear_angle_mismatch"].append(
            _make_warning(
                port_names,
                values=[port1.shear_angle, port2.shear_angle],
                message=f"{port_names[0]} has a shear angle but {port_names[1]} does not! Shear angle is {port1.shear_angle} deg",
            )
        )
    elif not port1.shear_angle and port2.shear_angle:
        warnings["shear_angle_mismatch"].append(
            _make_warning(
                port_names,
                values=[port1.shear_angle, port2.shear_angle],
                message=f"{port_names[1]} has a shear angle but {port_names[0]} does not! Shear angle is {port2.shear_angle} deg",
            )
        )
    elif port1.shear_angle:
        if (
            abs(difference_between_angles(port1.shear_angle, port2.shear_angle))
            > angle_tolerance
        ):
            warnings["shear_angle_mismatch"].append(
                _make_warning(
                    port_names,
                    values=[port1.shear_angle, port2.shear_angle],
                    message=f"Shear angle of {port_names[0]} and {port_names[1]} not equal. Difference of {abs(port1.shear_angle - port2.shear_angle)} deg",
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
                    message=f"{lower_port} was promoted to {top_port} but orientations do not match! Difference of {(abs(port1.orientation - port2.orientation))} deg",
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

    offset_mismatch = np.sqrt(np.sum(np.square(port2.center - port1.center)))
    if offset_mismatch > offset_tolerance:
        warnings["offset_mismatch"].append(
            _make_warning(
                port_names,
                values=[port1.center, port2.center],
                message=f"{port_names[0]} and {port_names[1]} are offset by {offset_mismatch} um",
            )
        )


def difference_between_angles(angle2: float, angle1: float):
    diff = angle2 - angle1
    while diff < 180:
        diff += 360
    while diff > 180:
        diff -= 360
    return diff


def _get_references_to_netlist(component: Component) -> List[ComponentReference]:
    from gdsfactory.cell import CACHE

    references = component.references
    if not references and "transformed_cell" in component.info:
        # expand transformed, flattened cells
        ref = component.settings.full["ref"]
        original_cell = CACHE[component.info["transformed_cell"]]
        references = [
            ComponentReference(
                original_cell,
                origin=ref["origin"],
                rotation=ref["rotation"],
                x_reflection=ref["x_reflection"],
            )
        ]
    return references


def get_netlist_recursive(
    component: Component,
    component_suffix: str = "",
    get_netlist_func: Callable = get_netlist,
    get_instance_name: Callable[..., str] = get_instance_name_from_alias,
    **kwargs,
) -> Dict[str, Any]:
    """Returns recursive netlist for a component and subcomponents.

    Args:
        component: to extract netlist.
        component_suffix: suffix to append to each component name.
            useful if to save and reload a back-annotated netlist.
        get_netlist_func: function to extract individual netlists.

    Keyword Args:
        full_settings: True returns all, false changed settings.
        tolerance: tolerance in nm to consider two ports connected.
        exclude_port_types: optional list of port types to exclude from netlisting.
        get_instance_name: function to get instance name.

    Returns:
        Dictionary of netlists, keyed by the name of each component.

    """
    all_netlists = {}

    # only components with references (subcomponents) warrant a netlist
    references = _get_references_to_netlist(component)

    if references:
        netlist = get_netlist_func(component, **kwargs)
        all_netlists[f"{component.name}{component_suffix}"] = netlist

        # for each reference, expand the netlist
        for ref in references:
            rcell = ref.parent
            grandchildren = get_netlist_recursive(
                component=rcell,
                component_suffix=component_suffix,
                get_netlist_func=get_netlist_func,
                **kwargs,
            )
            all_netlists.update(grandchildren)

            child_references = _get_references_to_netlist(ref.ref_cell)

            if child_references:
                inst_name = get_instance_name(component, ref)
                netlist_dict = {"component": f"{rcell.name}{component_suffix}"}
                if hasattr(rcell, "settings") and hasattr(rcell.settings, "full"):
                    netlist_dict.update(settings=rcell.settings.full)
                if hasattr(rcell, "info"):
                    netlist_dict.update(info=rcell.info)
                netlist["instances"][inst_name] = netlist_dict

    return all_netlists


def _demo_ring_single_array() -> None:
    import gdsfactory as gf

    c = gf.components.ring_single_array()
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
    print(c.get_netlist_yaml())


DEFAULT_CONNECTION_VALIDATORS = get_default_connection_validators()

DEFAULT_CRITICAL_CONNECTION_ERROR_TYPES = {
    "optical": ["width_mismatch", "shear_angle_mismatch", "orientation_mismatch"]
}


if __name__ == "__main__":
    # from pprint import pprint
    # from omegaconf import OmegaConf
    # import gdsfactory as gf
    # from gdsfactory.tests.test_component_from_yaml import sample_2x2_connections

    # c = gf.read.from_yaml(sample_2x2_connections)
    # c = gf.components.ring_single()
    # c.show(show_ports=True)
    # pprint(c.get_netlist())

    # n = c.get_netlist()
    # yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    # c2 = gf.read.from_yaml(yaml_str)
    # gf.show(c2)

    import gdsfactory as gf

    c = gf.components.mzi(delta_length=10)
    n = c.get_netlist()
    print(c.get_netlist_yaml())

    c = gf.read.from_yaml(c.get_netlist())
    c.show()

    # coupler_lengths = [10, 20, 30, 40]
    # coupler_gaps = [0.1, 0.2, 0.4, 0.5]
    # delta_lengths = [10, 100, 200]

    # c = gf.components.mzi_lattice(
    #     coupler_lengths=coupler_lengths,
    #     coupler_gaps=coupler_gaps,
    #     delta_lengths=delta_lengths,
    # )
    # n = c.get_netlist_recursive()

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
from typing import Any, Callable, Dict, List, Optional

import gdspy
import numpy as np
import omegaconf

from gdsfactory import Port
from gdsfactory.component import Component, ComponentReference
from gdsfactory.name import clean_name
from gdsfactory.pdk import get_layer
from gdsfactory.serialization import clean_value_json
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import LayerSpec


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
    tolerance: int = 1,
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
    tolerance: int = 1,
    exclude_port_types: Optional[List] = None,
    get_instance_name: Callable[..., str] = get_instance_name_from_alias,
) -> Dict[str, Any]:
    """From a component returns instances, connections and placements dict.

    Assumes that ports with same width, x, y are connected.

    Args:
        component: to extract netlist.
        full_settings: True returns all, false changed settings.
        layer_label: label to read instanceNames from (if any).
        tolerance: tolerance in nm to consider two ports connected.
        exclude_port_types: a list of port types to exclude from netlisting (optional)

    Returns:
        instances: Dict of instance name and settings.
        connections: Dict of Instance1Name,portName: Instace2Name,portName.
        placements: Dict of instance names and placements (x, y, rotation).
        port: Dict portName: ComponentName,port.
        name: name of component.

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

    for reference in component.references:
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
            for i in range(reference.rows):
                for j in range(reference.columns):
                    reference_name = base_reference_name + f"__{i + 1}_{j + 1}"
                    xj = x + j * reference.spacing[0]
                    yi = y + i * reference.spacing[1]
                    instances[reference_name] = instance
                    placements[reference_name] = dict(
                        x=xj,
                        y=yi,
                        rotation=int(reference.rotation or 0),
                        mirror=reference.x_reflection or 0,
                    )
                    parent_ports = c.ports
                    for parent_port_name in parent_ports:
                        top_name = f"{parent_port_name}_{i + 1}_{j + 1}"
                        lower_name = f"{reference_name},{parent_port_name}"
                        # a bit of a hack... get the top-level port for the ComponentArray, by our known naming convention. I hope no one renames these ports!
                        parent_port = component.ports[top_name]
                        name2port[top_name] = parent_port
                        name2port[lower_name] = parent_port
                        top_ports_list.add(top_name)
                        ports_by_type[parent_port.port_type].append(top_name)
                        ports_by_type[parent_port.port_type].append(lower_name)

    for port in ports:
        src = port.name
        name2port[src] = port
        top_ports_list.add(src)

    # lower level ports
    for reference in component.references:
        if isinstance(reference, gdspy.CellArray):
            pass
        else:
            for port in reference.ports.values():
                reference_name = get_instance_name(
                    component,
                    reference,
                )
                src = f"{reference_name},{port.name}"
                name2port[src] = port
                ports_by_type[port.port_type].append(src)

    for port_type, port_names in ports_by_type.items():
        connections_t, warnings_t = extract_connections(
            port_names, name2port, port_type
        )

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
    return dict(
        connections=connections_sorted,
        instances=instances_sorted,
        placements=placements_sorted,
        ports=top_ports,
        name=component.name,
    )


def extract_connections(port_names: List[str], ports: Dict[str, Port], port_type: str):
    if port_type == "optical":
        return extract_optical_connections(port_names, ports)
    elif port_type == "electrical":
        return extract_electrical_connections(port_names, ports)


def extract_optical_connections(
    port_names: List[str],
    ports: Dict[str, Port],
    raise_error_for_warnings=(
        "width_mismatch",
        "shear_angle_mismatch",
        "orientation_mismatch",
    ),
):
    angle_tolerance = 1  # degrees
    by_xy_1nm = defaultdict(list)
    warnings = defaultdict(list)

    for port_name in port_names:
        port = ports[port_name]
        by_xy_1nm[tuple(np.round(port.center, 3))].append(port_name)

    unconnected_port_names = []
    connections = []

    for xy, ports_at_xy in by_xy_1nm.items():
        if len(ports_at_xy) == 1:
            unconnected_port_names.append(ports_at_xy[0])
            warnings["unconnected_ports"].append(ports_at_xy[0])
        elif len(ports_at_xy) == 2:
            port1 = ports[ports_at_xy[0]]
            port2 = ports[ports_at_xy[1]]

            is_top_level = [("," not in pname) for pname in ports_at_xy]
            if all(is_top_level):
                raise ValueError(
                    f"Two top-level ports appear to be connected: {ports_at_xy}"
                )

            # assert no angle mismatch
            # assert no width mismatch
            if port1.width != port2.width:
                warnings["width_mismatch"].append(ports_at_xy)
            if port1.shear_angle != port2.shear_angle:
                warnings["shear_angle_mismatch"].append(ports_at_xy)

            if any(is_top_level):
                if port1.orientation != port2.orientation:
                    warnings["orientation_mismatch"].append(ports_at_xy)
            elif (
                abs(abs(port1.orientation - port2.orientation) - 180) > angle_tolerance
            ):
                warnings["orientation_mismatch"].append(ports_at_xy)
            connections.append(ports_at_xy)
        else:
            warnings["multiple_connections"].append(ports_at_xy)
            raise ValueError(f"Found multiple connections at {xy}:{ports_at_xy}")

    critical_warnings = {
        w: warnings[w] for w in raise_error_for_warnings if warnings[w]
    }

    if critical_warnings:
        raise ValueError(
            f"Found warnings while extracting netlist: {critical_warnings}"
        )
    return connections, warnings


def extract_electrical_connections(port_names: List[str], ports: Dict[str, Port]):
    # for port_name in port_names:
    #     port = ports[port_name]
    raise NotImplementedError()


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
        layer_label: label to read instanceNames from (if any).
        tolerance: tolerance in nm to consider two ports connected.

    Returns:
        Dictionary of netlists, keyed by the name of each component.

    """
    all_netlists = {}

    # only components with references (subcomponents) warrant a netlist
    if component.references:
        netlist = get_netlist_func(component, **kwargs)
        all_netlists[f"{component.name}{component_suffix}"] = netlist

        # for each reference, expand the netlist
        for ref in component.references:
            rcell = ref.parent
            grandchildren = get_netlist_recursive(
                component=rcell,
                component_suffix=component_suffix,
                get_netlist_func=get_netlist_func,
                **kwargs,
            )
            all_netlists.update(grandchildren)
            if ref.ref_cell.references:
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
    print("\n".join(n["instances"].keys()))

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

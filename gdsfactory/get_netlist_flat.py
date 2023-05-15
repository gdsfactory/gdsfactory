from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Dict, List
from gdsfactory.component import Component
import gdsfactory as gf
from gdsfactory.get_netlist import get_netlist_recursive


def get_netlist_flat(
    component: Component,
    **kwargs,
) -> Dict[str, Any]:
    """Parses a recursive netlist for a component as if it was a single netlist \
            with its lowest-level instances.

    Procedure:
        - Recursively parse the recursive dict to generate a unique list of all instances **even if they are reused**
          Each entry is formatted as [(netlist, instance),...], and defines a unique flat_name = {top}{hierarchy_delimiter}{instance}{hierarchy_delimiter}{instance}{hierarchy_delimiter}...
        - Populate the flat netlist dict with similar entries as regular gdsfactory netlists:
            - instances: component, info, and settings from the original netlist, but keyed with flat_name
            - placements: For instance corresponding to each flat_name, accumulate placements across the hierarchy
            - ports and connections: each unique port of each flat_name instance is uniquely named as flat_name,port_name
                                     for each flat_name,port_name, starting at lowest hierarchy level:
                                        - list connections at that level (finding the proper other flat_name,port_name if non-leaf level)
                                        - if possible, maps the port to a port of that instance's netlist, and repeat at a higher level
                                        - if the top level is reached, assign that flat_name,port_name to a top-level component port instead
                    the returned ports dict has top level component portname: flat_name,port_name mappings
                    the returned connections dict is sorted and flattened into a minimal (flat_name,port_name)_1: [(flat_name,port_name)] key: value pairs
                        If allow_multiple flag in get_netlist is True, the values will be lists (to support multiple connections)
            - name: top_level_component name

    Args:
        component: to extract flat netlist.

    Keyword Args:
        component_suffix: suffix to append to each component name.
            useful if to save and reload a back-annotated netlist.
        get_netlist_func: function to extract individual netlists.
        full_settings: True returns all, false changed settings.
        tolerance: tolerance in nm to consider two ports connected.
        exclude_port_types: optional list of port types to exclude from netlisting.
        get_instance_name: function to get instance name.

    Returns:
        instances: Dict of instance name and settings.
        connections: Dict of Instance1Name,portName: Instance2Name,portName.
        placements: Dict of instance names and placements (x, y, rotation).
        port: Dict portName: ComponentName,port.
        name: name of component.
        warnings: warning messages (disconnected pins).
    """
    recursive_netlist = get_netlist_recursive(component, **kwargs)
    top_level_name = component.name
    hierarchical_instances = _flatten_hierarchy(top_level_name, recursive_netlist)
    connections = {}
    ports = {}
    placements = {}
    instances = {}
    for hierarchical_instance in hierarchical_instances:
        c, p = _map_connections_ports(
            hierarchical_instance, top_level_name, recursive_netlist
        )
        for key, value in c.items():
            if len(value) != 0:
                connections[key] = value
        for key, value in p.items():
            if len(value) != 0:
                ports[key] = value
        placements.update(
            _accumulate_placements(hierarchical_instance, recursive_netlist)
        )
        instances.update(_get_instance_info(hierarchical_instance, recursive_netlist))

    return {
        "connections": connections,
        "placements": placements,
        "instances": instances,
        "ports": ports,
        "name": recursive_netlist[top_level_name]["name"],
    }


def _flat_name(
    hierarchy,
    hierarchy_delimiter: str = "~",
):
    """Returns a unique name for the instance from its hierarchy."""
    name = ""
    for _, instance in hierarchy[:-1]:
        name += instance
        name += hierarchy_delimiter
    name += hierarchy[-1][1]
    return name


def _get_instance_info(hierarchy, recursive_netlist):
    """Returns instance info from its instance data."""
    return {
        _flat_name(hierarchy): recursive_netlist[hierarchy[-2][0]]["instances"][
            hierarchy[-1][1]
        ]
    }


def _get_leaf(
    instance_port,
    netlist,
    flat_name,
    all_netlists,
    hierarchy_delimiter: str = "~",
):
    """Given a instance, port and its netlist, maps ports down to the lowest hierarchy level."""
    instance, port = instance_port.split(",")
    netlist = all_netlists[netlist]["instances"][instance]["component"]
    found_lower_level = netlist in all_netlists.keys()
    if found_lower_level:
        instance_port = all_netlists[netlist]["ports"][port]
        flat_name = f"{flat_name}{hierarchy_delimiter}{instance}"
        found_lower_level, flat_name = _get_leaf(
            instance_port, netlist, flat_name, all_netlists
        )
    else:
        found_lower_level = False
        flat_name = f"{flat_name}{hierarchy_delimiter}{instance_port}"
    return found_lower_level, flat_name


def _lateral_map(
    local_leaf_port: str,
    all_netlists: Dict[str, Any],
    higher_component: str,
    level: int,
    hierarchy,
    hierarchy_delimiter: str = "~",
):
    """Returns connected ports at this hierarchical level."""
    lateral_equivalencies = []
    for port1, port2 in all_netlists[higher_component]["connections"].items():
        flat_name_prefix = _flat_name(hierarchy[: len(hierarchy) - level - 1])
        if local_leaf_port == port1 and local_leaf_port != port2:
            lateral_equivalencies.append(
                _get_leaf(port2, higher_component, flat_name_prefix, all_netlists)[1]
            )
        elif local_leaf_port == port2 and local_leaf_port != port1:
            lateral_equivalencies.append(
                _get_leaf(port1, higher_component, flat_name_prefix, all_netlists)[1]
            )
    return lateral_equivalencies


def _map_connections_ports(
    hierarchy,
    top_name,
    all_netlists,
    hierarchy_delimiter: str = "~",
):
    """Returns all nodes of interest across the flat recursive netlist."""
    connections = {}
    ports = {}

    # Starting point is ports of the leaf instance
    leaf_instance = hierarchy[-1][1]
    leaf_instance_ports = list(gf.get_component(hierarchy[-1][0]).ports.keys())

    for leaf_portname in leaf_instance_ports:
        current_connections = []
        current_ports = []
        local_leaf_port = f"{leaf_instance},{leaf_portname}"
        for level, (higher_component, higher_instance) in enumerate(
            hierarchy[:-1][::-1]
        ):
            # Identify connected node (with flattened names)
            current_connections.extend(
                _lateral_map(
                    local_leaf_port,
                    all_netlists,
                    higher_component,
                    level,
                    hierarchy,
                    hierarchy_delimiter,
                )
            )
            # Traverse up to higher level
            found_higher_level = False
            for portname, port in all_netlists[higher_component]["ports"].items():
                if local_leaf_port == port:
                    local_leaf_port = f"{higher_instance},{portname}"
                    found_higher_level = True
                    # If there is a port at top level, map to that as well
                    if higher_instance == top_name:
                        current_ports.append(local_leaf_port)
            if not found_higher_level:
                break
        flat_leaf_port = f"{_flat_name(hierarchy)},{leaf_portname}"
        connections[flat_leaf_port] = current_connections
        ports[flat_leaf_port] = current_ports

    return connections, ports


def _accumulate_placements(
    hierarchy,
    all_netlists,
):
    """Iterate through hierarchy tuples, accumulating placement information."""
    placements = {key: 0 for key in ["x", "y", "mirror", "rotation"]}
    for (higher_component, _higher_instance), (_lower_component, lower_instance) in zip(
        hierarchy[:-1], hierarchy[1:]
    ):
        for key in ["x", "y", "mirror", "rotation"]:
            placements[key] += all_netlists[higher_component]["placements"][
                lower_instance
            ][key]

    return {_flat_name(hierarchy): placements}


def _flatten_str_list(xs):
    """Flatten nested list of strings to list of strings."""
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from _flatten_str_list(x)
        else:
            yield x


def _flatten_hierarchy(
    netlist_name: str,
    all_netlists: List[Dict[str, any]],
    hierarchy_delimiter: str = "~",
    component_instance_delimiter: str = ";",
):
    """Converts _flatten_netlist output str's to list of hierarchical (component, instance) tuples."""

    instance_dict = _flatten_hierarchy_recurse(
        netlist_name=netlist_name,
        all_netlists=all_netlists,
        hierarchy_delimiter=hierarchy_delimiter,
        component_instance_delimiter=component_instance_delimiter,
    )

    hierarchies_lists = []
    for hierarchy in _flatten_str_list(list(instance_dict.keys())):
        hierarchy_list = []
        levels = hierarchy.split(hierarchy_delimiter)
        for level in levels:
            components, instance = level.split(component_instance_delimiter)
            hierarchy_list.append((components, instance))
        hierarchies_lists.append(hierarchy_list)

    return hierarchies_lists


def _flatten_hierarchy_recurse(
    netlist_name: str,
    all_netlists: List[Dict[str, any]],
    hierarchy: str = None,
    hierarchy_delimiter: str = "~",
    component_instance_delimiter: str = ";",
) -> Dict[str, Any]:
    """Flattens the provided recursive netlist by recursively updating a list.

    Args:
        netlist_name: netlist entry to flatten
        all_netlists: list of all possible netlists (output of get_netlist_recursive)
        hierarchy: str to append to netlist_name
        hierarchy_delimiter: str to separate hierarchy levels in flattened keys
        component_instance_delimiter: str to separate instance name from component name for netlist reconstruction

    Returns:
        str of ...{hierarchy_delimiter}{component}{component_instance_delimiter}{instance}{hierarchy_delimiter} used to flatten the netlist
    """
    instance_dict = {}
    if hierarchy is None:
        top_component = all_netlists[netlist_name]["name"]
        hierarchy = f"{top_component}{component_instance_delimiter}{netlist_name}"
    for instance_name, instance in all_netlists[netlist_name]["instances"].items():
        component_name = instance["component"]
        hierarchy_str = f"{hierarchy}{hierarchy_delimiter}{component_name}{component_instance_delimiter}{instance_name}"
        if component_name not in all_netlists.keys():
            instance_dict[hierarchy_str] = True  # Done for this leaf
        else:
            instance_dict.update(
                _flatten_hierarchy_recurse(
                    component_name,
                    all_netlists,
                    hierarchy_str,
                )
            )

    return instance_dict


if __name__ == "__main__":
    import pprint

    coupler_lengths = [10, 20, 30, 40]
    coupler_gaps = [0.1, 0.2, 0.4, 0.5]
    delta_lengths = [10, 100, 200]

    c = gf.components.mzi_lattice(
        coupler_lengths=coupler_lengths,
        coupler_gaps=coupler_gaps,
        delta_lengths=delta_lengths,
    )
    flat_netlist = get_netlist_flat(c)
    print("")
    print("FLAT NETLIST")
    print("")
    pprint.pprint(flat_netlist)

    # """
    # Testing electrical netlist w/ identical component references
    # """
    # # Define compound component
    # series_resistors = gf.Component("seriesResistors")
    # rseries1 = series_resistors << gf.get_component(
    #     gf.components.resistance_sheet, width=20, ohms_per_square=20
    # )
    # rseries2 = series_resistors << gf.get_component(
    #     gf.components.resistance_sheet, width=20, ohms_per_square=20
    # )
    # rseries1.connect("pad2", rseries2.ports["pad1"])
    # series_resistors.add_port("pad1", port=rseries1.ports["pad1"])
    # series_resistors.add_port("pad2", port=rseries2.ports["pad2"])

    # # Increase hierarchy levels more
    # double_series_resistors = gf.Component("double_seriesResistors")
    # rseries1 = double_series_resistors << gf.get_component(series_resistors)
    # rseries2 = double_series_resistors << gf.get_component(series_resistors)
    # rseries1.connect("pad2", rseries2.ports["pad1"])
    # double_series_resistors.add_port("pad1", port=rseries1.ports["pad1"])
    # double_series_resistors.add_port("pad2", port=rseries2.ports["pad2"])

    # # Define top-level component
    # vdiv = gf.Component("voltageDivider")
    # r1 = vdiv << double_series_resistors
    # r2 = vdiv << series_resistors
    # r3 = (
    #     vdiv
    #     << gf.get_component(
    #         gf.components.resistance_sheet, width=20, ohms_per_square=20
    #     ).rotate()
    # )
    # r4 = vdiv << gf.get_component(
    #     gf.components.resistance_sheet, width=20, ohms_per_square=20
    # )

    # r1.connect("pad2", r2.ports["pad1"])
    # r3.connect("pad1", r2.ports["pad1"], preserve_orientation=True)
    # r4.connect("pad1", r3.ports["pad2"], preserve_orientation=True)

    # vdiv.add_port("gnd1", port=r2.ports["pad2"])
    # vdiv.add_port("gnd2", port=r4.ports["pad2"])
    # vdiv.add_port("vsig", port=r1.ports["pad1"])
    # vdiv.show(show_ports=True)

    # recursive_netlist = get_netlist_recursive(vdiv, allow_multiple=True)
    # import pprint

    # print("RECURSIVE NETLIST")
    # print("")
    # pprint.pprint(recursive_netlist)
    # print("")
    # print("OPERATION")
    # print("")

    # flat_netlist = get_netlist_flat(vdiv, allow_multiple=True)
    # print("")
    # print("FLAT NETLIST")
    # print("")
    # pprint.pprint(flat_netlist)

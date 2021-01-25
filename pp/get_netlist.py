"""Simpler netlist.

.. code:: yaml

    connections:
        - coupler,N0:bendLeft,W0
        - coupler,N1:bendRight,N0
        - bednLeft,N0:waveguide,W0
        - bendRight,N0:waveguide,E0

    ports:
        - coupler,E0
        - coupler,W0

"""

from typing import Dict, Tuple

from pp.component import Component, ComponentReference
from pp.drc import snap_to_1nm_grid
from pp.layers import LAYER


def get_instance_name(
    component: Component,
    reference: ComponentReference,
    layer_label: Tuple[int, int] = LAYER.LABEL_INSTANCE,
) -> str:
    """Takes a component names the instance based on its XY location or a label in layer_label
    Loop over references and find the reference under and associate reference with instance label
    map instance names to references
    Check if it has a instance name label and return the instance name from the label

    Args:
        component: with labels
        reference: reference that needs naming
        layer_label: layer of the label (ignores layer_label[1]). Phidl ignores purpose of labels.
    """

    x = snap_to_1nm_grid(reference.x)
    y = snap_to_1nm_grid(reference.y)
    labels = component.labels

    # default instance name follows componetName_x_y
    text = f"{reference.parent.name}_{x}_{y}"
    # text = f"{reference.parent.name}_X{int(x)}_Y{int(y)}"
    # text = f"{reference.parent.name}_{reference.uid}"

    # try to get the instance name from a label
    for label in labels:
        xl = snap_to_1nm_grid(label.x)
        yl = snap_to_1nm_grid(label.y)
        if x == xl and y == yl and label.layer == layer_label[0]:
            # print(label.text, xl, yl, x, y)
            return label.text

    return text


def get_netlist(
    component: Component,
    full_settings: bool = False,
    layer_label: Tuple[int, int] = LAYER.LABEL_INSTANCE,
) -> Dict[str, Dict]:
    """From a component returns instances and placements dicts.
    it assumes that ports with same x,y are connected.

    Args:
        full_settings: True returns all settings, false only the ones that have changed
        layer_label: label to read instanceNames from (if any)

    Returns:
        connections: Dict of Instance1Name,portName: Instace2Name,portName
        instances: Dict of instances and settings
        placements: Dict of instances and placements (x, y, rotation)
        port: Dict portName: CompoentName,port
        name: name of component

    """
    placements = {}
    instances = {}
    connections = {}
    top_ports = {}

    for reference in component.references:
        c = reference.parent
        origin = snap_to_1nm_grid(reference.origin)
        x = snap_to_1nm_grid(origin[0])
        y = snap_to_1nm_grid(origin[1])
        reference_name = get_instance_name(
            component, reference, layer_label=layer_label
        )

        settings = c.get_settings(full_settings=full_settings)
        instances[reference_name] = dict(
            component=c.function_name, settings=settings["settings"],
        )
        placements[reference_name] = dict(
            x=x, y=y, rotation=int(reference.rotation), mirror=reference.x_reflection,
        )

    # store where ports are located
    name2port = {}

    # Initialize a dict of port locations to Instance1Name,PortNames
    port_locations = {}

    # TOP level ports
    ports = component.get_ports(depth=0)
    top_ports_list = set()
    for port in ports:
        src = port.name
        name2port[src] = port
        top_ports_list.add(src)

    # lower level ports
    for reference in component.references:
        for port in reference.ports.values():
            reference_name = get_instance_name(
                component, reference, layer_label=layer_label
            )
            src = f"{reference_name},{port.name}"
            name2port[src] = port

    # build connectivity port_locations = Dict[Tuple(x,y), set of portNames]
    for name, port in name2port.items():
        xy = snap_to_1nm_grid((port.x, port.y))
        if xy not in port_locations:
            port_locations[xy] = set()
        port_locations[xy].add(name)

    for xy, names_set in port_locations.items():
        if len(names_set) > 2:
            raise ValueError(f"more than 2 connections at {xy} {list(names_set)}")
        if len(names_set) == 2:
            names_list = list(names_set)
            src = names_list[0]
            dst = names_list[1]
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


def demo_ring_single_array():
    import pp

    c = pp.c.ring_single_array()
    c.get_netlist()


def demo_mzi_lattice():
    import pp

    coupler_lengths = [10, 20, 30, 40]
    coupler_gaps = [0.1, 0.2, 0.4, 0.5]
    delta_lengths = [10, 100, 200]

    c = pp.c.mzi_lattice(
        coupler_lengths=coupler_lengths,
        coupler_gaps=coupler_gaps,
        delta_lengths=delta_lengths,
    )
    c.get_netlist()
    print(c.get_netlist_yaml())


if __name__ == "__main__":
    from pprint import pprint

    from omegaconf import OmegaConf

    import pp
    from pp.test_component_from_yaml import sample_2x2_connections

    c = pp.component_from_yaml(sample_2x2_connections)
    pp.show(c)
    pprint(c.get_netlist())

    n = c.get_netlist()
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c2 = pp.component_from_yaml(yaml_str)
    pp.show(c2)

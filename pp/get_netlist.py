"""Simpler netlist.

FIXME. Would be nice to go back from netlist to layout


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

from pp.drc import snap_to_1nm_grid
from pp.layers import LAYER


def get_instance_name(component, reference, layer_label=LAYER.LABEL_INSTANCE):
    """Takes a component
    Loop over references and find the reference under and associate reference with instance label
    map instance names to references
    """
    # TODO: check if it has a instance name label
    # return f"{reference.parent.name}_{reference.uid}"
    x = snap_to_1nm_grid(reference.x)
    y = snap_to_1nm_grid(reference.y)
    return f"{reference.parent.name}_{x}_{y}"


def get_netlist(component, full_settings=False):
    """From a component returns instances and placements dicts.
    it assumes that ports with same x,y are connected.

    Args:
        full_settings: True returns all settings, false only the ones that have changed

    Returns:
        connections: Dict of Instance1Name,portName: Instace2Name,portName
        instances: Dict of instances and settings
        placements: Dict of instances and placements (x, y, rotation)

    """
    placements = {}
    instances = {}
    connections = {}
    top_ports = {}

    for reference in component.references:
        c = reference.parent
        x = snap_to_1nm_grid(reference.x)
        y = snap_to_1nm_grid(reference.y)
        reference_name = get_instance_name(component, reference)
        settings = c.get_settings(full_settings=full_settings)
        instances[reference_name] = dict(
            component=c.function_name, settings=settings["settings"]
        )
        # dx = snap_to_1nm_grid(reference.x - component.x)
        # dy = snap_to_1nm_grid(reference.y - component.y)
        placements[reference_name] = dict(x=x, y=y, rotation=int(reference.rotation))

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
            reference_name = get_instance_name(component, reference)
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
                connections[src] = dst

    # connections_sorted = connections
    # track connections starting from an arbitrary port (src0)
    # connections are defined as sourceInstance,port: destinationInstance,port
    # find other connections to destinationInstance,port2
    # connections_sorted = {}
    # while connections:
    #     src0 = list(connections.keys())[0]
    #     dst0 = connections.pop(src0)
    #     connections_sorted[src0] = dst0
    #     next_instance_name = dst0.split(',')[0]
    #     remanining_connections = list(connections.keys())

    #     for src in remanining_connections:
    #         dst = connections[src]
    #         src_instance_name = src.split(',')[0]
    #         dst_instance_name = dst.split(',')[0]

    #         # next
    #         if src_instance_name == next_instance_name:
    #             connections.pop(src)
    #             connections_sorted[dst] = dst0
    #             dst0 = dst
    #             continue
    #         elif dst_instance_name == next_instance_name:
    #             connections.pop(src)
    #             connections_sorted[src] = dst0
    #             dst0 = src
    #             continue

    connections_sorted = {k: connections[k] for k in sorted(list(connections.keys()))}
    placements_sorted = {k: placements[k] for k in sorted(list(placements.keys()))}
    instances_sorted = {k: instances[k] for k in sorted(list(instances.keys()))}
    return connections_sorted, instances_sorted, placements_sorted, top_ports


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
    # test_mzi_lattice()
    # import matplotlib.pyplot as plt
    from omegaconf import OmegaConf

    import pp

    # c = pp.c.ring_single_array()
    # pp.show(c)

    c = pp.c.ring_single()
    ports = c.get_ports(depth=0)
    pp.show(c)

    connections, instances, placements, ports = get_netlist(c)
    # connections, instances, placements = get_netlist(c.references[0].parent)
    print(connections)
    # print(ports)
    # print(instances)

    n = c.get_netlist()
    # n.pop('connections')
    # n.pop('placements')

    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    # print(yaml_str)
    c2 = pp.component_from_yaml(yaml_str)
    pp.show(c2)

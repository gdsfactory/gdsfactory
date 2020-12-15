"""Simpler netlist. FIXME. Still work in progress


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
    return reference.uid


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

    for reference in component.references:
        c = reference.parent
        x = snap_to_1nm_grid(reference.x)
        y = snap_to_1nm_grid(reference.y)
        reference_name = f"{c.name}_{int(x)}_{int(y)}"
        settings = c.get_settings(full_settings=full_settings)
        instances[reference_name] = dict(component=c.function_name, settings=settings)
        placements[reference_name] = dict(x=x, y=y, rotation=int(reference.rotation))

    # get top level ports are not connected
    ports = component.get_ports(depth=0)
    # for port in ports:
    #     connections[f'TOP,{port.name}'] = None

    # Initialize a dict of port locations to Instance1Name,PortNames
    port_locations = {snap_to_1nm_grid((port.x, port.y)): set() for port in ports}

    for reference in component.references:
        for port in reference.ports:
            instance_name = get_instance_name(component, reference)
            src = f"{instance_name},{port.name}"
            xy = snap_to_1nm_grid((port.x, port.y))
            assert (
                xy in port_locations
            ), f"{xy} for {port.name} {c.name} in  not in {port_locations}"
            src_list = port_locations[xy]

            # first time encountered ports append to port_locations dict
            if len(src_list) == 0:
                src_list.add(src)
            else:
                for src2 in src_list:
                    connections[src2] = src
                    # connections[src] = src2

    placements_sorted = {k: placements[k] for k in sorted(list(placements.keys()))}
    instances_sorted = {k: instances[k] for k in sorted(list(instances.keys()))}
    return connections, instances_sorted, placements_sorted


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
    import pp

    # c = pp.c.ring_single_array()
    # pp.show(c)

    c = pp.c.ring_single()
    pp.show(c)

    connections, instances, placements = get_netlist(c)
    # connections, instances, placements = get_netlist(c.references[0].parent)
    print(connections)

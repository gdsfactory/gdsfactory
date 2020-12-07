from pp.drc import snap_to_1nm_grid


def recurse_references(
    component,
    instances=None,
    placements=None,
    connections=None,
    port_locations=None,
    dx: float = 0.0,
    dy: float = 0.0,
    recursive=True,
    level: int = 0,
):
    """From a component returns instances and placements dicts.
    it assumes that ports with same x,y are connected.
    Ensures that connections are at the same level of hierarchy.

    Args:
        component: to recurse
        instances: instance_name to settings dict. Instances are name by ComponentName.x.y
        placements: instance_name to x,y,rotation dict
        connections: instance_name_src,portName: instance_name_dst,portName
        port_locations: dict((x,y): set([referenceName, Port]))
        dx: port displacement in x (for recursice case)
        dy: port displacement in y (for recursive case)
        recursive: goes down the hierarchy
        level: current level of the hierarchy (0: Top level, 1: first level ...)

    Returns:
        connections: Dict of Instance1Name,portName: Instace2Name,portName
        instances: Dict of instances and settings
        placements: Dict of instances and placements (x, y, rotation)

    """
    placements = placements or {}
    instances = instances or {}
    connections = connections or set()
    port_locations = port_locations or {
        snap_to_1nm_grid((port.x, port.y)): set() for port in component.get_ports()
    }

    for r in component.references:
        c = r.parent
        x = snap_to_1nm_grid(r.x + dx)
        y = snap_to_1nm_grid(r.y + dy)
        reference_name = f"{c.name}_{int(x)}_{int(y)}"
        settings = c.get_settings()
        instances[reference_name] = dict(component=c.function_name, settings=settings)
        placements[reference_name] = dict(x=x, y=y, rotation=int(r.rotation))
        for port in r.get_ports_list():
            src = f"{reference_name},{port.name}"
            xy = snap_to_1nm_grid((port.x + dx, port.y + dy))
            assert (
                xy in port_locations
            ), f"{xy} for {port.name} {c.name} in level {level} not in {port_locations}"
            src_list = port_locations[xy]
            if len(src_list) > 0:
                for src2 in src_list:
                    connections.add((src, src2))
                    print(src2)
            else:  # first time that port appears
                src_list.add(src)

    if recursive:
        for r in component.references:
            c = r.parent
            dx = r.x - c.x
            dy = r.y - c.y
            # print(level, c.name, r.x, dx, c.x)
            if len(c.references) > 0:
                c2, i2, p2 = recurse_references(
                    component=c,
                    instances=instances,
                    placements=placements,
                    connections=connections,
                    dx=dx,
                    dy=dy,
                    port_locations=port_locations,
                    level=level + 1,
                )
                placements.update(p2)
                instances.update(i2)
                connections.union(c2)

    placements_sorted = {k: placements[k] for k in sorted(list(placements.keys()))}
    instances_sorted = {k: instances[k] for k in sorted(list(instances.keys()))}
    return connections, instances_sorted, placements_sorted


def test_ring_single_array():
    import pp

    c = pp.c.ring_single_array()
    c.get_netlist()


def test_mzi_lattice():
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


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    import pp

    c = pp.c.ring_single_array()
    pp.show(c)
    c.plot_netlist()

    x, i, p = recurse_references(c)
    # plt.show()

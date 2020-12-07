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
    connections = connections or {}
    port_locations = port_locations or {
        snap_to_1nm_grid((port.x, port.y)): [] for port in component.get_ports()
    }

    level_name = f"{level}_{component.name}"
    connections[level_name] = {}

    for r in component.references:
        c = r.parent
        x = snap_to_1nm_grid(r.x + dx)
        y = snap_to_1nm_grid(r.y + dy)
        reference_name = f"{c.name}_{int(x)}_{int(y)}"
        settings = c.get_settings()
        instances[reference_name] = dict(component=c.function_name, settings=settings)
        placements[reference_name] = dict(x=x, y=y, rotation=int(r.rotation),)
        for port in r.get_ports_list():
            src = f"{reference_name},{port.name}"
            xy = snap_to_1nm_grid((port.x + dx, port.y + dy))
            assert (
                xy in port_locations
            ), f"{xy} for {c.name} in level {level} not in {port_locations}"
            src_list = port_locations[xy]
            if len(src_list) > 0:
                for src2 in src_list:
                    connections[level_name][src2] = src
                    connections[level_name][src] = src2
            else:
                src_list.append(src)

    if recursive:
        for r in component.references:
            c = r.parent
            x = snap_to_1nm_grid(r.x + dx)
            y = snap_to_1nm_grid(r.y + dy)
            if len(c.references) > 0:
                c2, i2, p2 = recurse_references(
                    component=c,
                    instances=instances,
                    placements=placements,
                    connections=connections,
                    dx=x - c.x,
                    dy=y - c.y,
                    port_locations=port_locations,
                    level=level + 1,
                )
                placements.update(p2)
                instances.update(i2)
                connections.update(c2)

    placements_sorted = {k: placements[k] for k in sorted(list(placements.keys()))}
    instances_sorted = {k: instances[k] for k in sorted(list(instances.keys()))}
    return connections, instances_sorted, placements_sorted


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pp

    c = pp.c.ring_single_array()
    c.plot_netlist()
    pp.show(c)

    c, i, p = recurse_references(c)
    plt.show()

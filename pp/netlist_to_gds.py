import numpy as np
from pp.component import Component
from pp.component import ComponentReference
from pp.ports.utils import select_electrical_ports

IDENTITY = (0, False)
R90 = (90, False)
R180 = (180, False)
R270 = (270, False)
mirror_x = (0, True)
mirror_y = (180, True)
mirror_x270 = (270, True)
mirror_x90 = (90, True)

TRANSFORMATION_MAP = {
    IDENTITY: "None",
    R90: "R90",
    R180: "R180",
    R270: "R270",
    mirror_y: "mirror_y",
    mirror_x: "mirror_x",
    mirror_x270: "mirror_x270",
    mirror_x90: "mirror_x90",
}

STR_TO_TRANSFORMATION_MAP = {v: k for k, v in TRANSFORMATION_MAP.items()}


def get_elec_ports_from_component_names(component, names=[]):
    """
    Args
        component <pp.Component>; should have component.info["components"]
    """
    e_ports = {}

    for name in names:
        _ports = {
            "{}_{}".format(name, p.name): p
            for p in select_electrical_ports(
                component.info["components"][name]
            ).values()
        }
        e_ports.update(_ports)

    # update port names
    for pname, p in e_ports.items():
        p.name = pname

    return e_ports


def gen_sref(component, transformation_name, port_name, position):
    """
    """

    if transformation_name not in TRANSFORMATION_MAP.values():
        raise ValueError(
            "{} is not a valid transformation({})".format(
                transformation_name, list(TRANSFORMATION_MAP.values())
            )
        )
    rotation_angle, x_reflection = STR_TO_TRANSFORMATION_MAP[transformation_name]
    position = np.array(position)

    if port_name is None:
        port_position = np.array([0, 0])
    else:
        if port_name not in component.ports.keys():
            raise ValueError(
                "{} port name not in {} ports ({})".format(
                    port_name, component.name, component.ports.keys()
                )
            )
        port_position = component.ports[port_name].midpoint

    device_ref = ComponentReference(device=component, origin=(0, 0))

    if x_reflection:  # Vertical mirror: Reflection across x-axis
        y0 = port_position[1]
        device_ref.reflect(p1=(0, y0), p2=(1, y0))

    if rotation_angle != 0:
        device_ref.rotate(rotation_angle, center=port_position)

    translation = np.array(position - port_position)

    device_ref.move(destination=translation)

    return device_ref


def netlist_to_component(components, connections, ports_map, position=(0, 0)):
    """
    Args:
        components:
            list of (component_id <str>, component <Component>, transform <tuple>)

        connections:
            list of (component_id1, port_name1, component_id2, port_name2)
            Has to be ordered such that any component refered to in the connections
            has already been placed (except for the first component)

        ports_map:
            {port_name: (component_id, port_name)}

    Returns: component
    the component has component.netlist

    [
        {
            "name": "CP2x2",
            "rank": 0,
            "ports": ["in1", "in2", "out1", "out2"],
            "type": "CPBIDIR",
            "settings": {"R": 0.5},
        },
        {
            "name": "WG_CAVITY",
            "rank": 0,
            "ports": ["in", "out"],
            "type": "WG",
            "settings": {"length": 50, "loss_dB_m": 60000, "width": 0.5},
        },
        {
            "name": "ring_resonator",
            "rank": 1,
            "type": "COMPOUND",
            "settings": {},
            "components": [
                "CP1, CP2x2, None, None, 0.0, 0.0",
                "WG1, WG_CAVITY, None, None, 0.0, 0.0",
            ],
            "connections": ["CP1, out1, WG1, in", "WG1, out, CP1, in1"],
            "ports": {"in1": "CP1, in2", "out1": "CP1, out2"},
        },
    ]

    mirror, rotation, x, y
    """
    if len(connections) == 0:
        raise ValueError(
            "Error number of connections", len(connections), len(components)
        )

    component_id, cmp_port, _, _ = connections[0]

    component, transform_name = components[component_id]

    # First component reference
    sref_start = gen_sref(component, transform_name, cmp_port, position)
    cmp_name_to_sref = {component_id: sref_start}

    # Iterate over all connections: create and place components
    for cmp1_name, port1, cmp2_name, port2 in connections:
        if cmp1_name not in cmp_name_to_sref:
            cmp1_name, port1, cmp2_name, port2 = cmp2_name, port2, cmp1_name, port1

        if cmp2_name not in cmp_name_to_sref:
            component, transform_name = components[cmp2_name]

            _ref = cmp_name_to_sref[cmp1_name]
            try:
                position = _ref.ports[port1].midpoint
            except:
                print("{} has not port {}".format(cmp1_name, port1))

            sref = gen_sref(component, transform_name, port2, position)

            cmp_name_to_sref[cmp2_name] = sref

    c = Component()
    c.add(list(cmp_name_to_sref.values()))
    for port_name, (cmp_id, internal_port_name) in ports_map.items():
        c.add_port(port_name, port=cmp_name_to_sref[cmp_id].ports[internal_port_name])

    # Set aliases
    for cmp_id, ref in cmp_name_to_sref.items():
        c[cmp_id] = ref

    c.info["components"] = cmp_name_to_sref
    # c.components = components
    # c.connections = connections
    # c.ports_map = ports_map
    # c.cells = {
    #     cname: c for cname, (c, transf) in components.items()
    # }  # "CP1": (cpl, "None"),

    # add leaf cells to netlist
    netlist = []
    for name, (component, _) in components.items():
        netlist.append(
            dict(
                name=name,
                rank=0,
                ports=list(component.ports.keys()),
                settings=component.settings,
            )
        )
    # add the compound cell to the netlist
    netlist.append(
        dict(
            name="component_name",
            rank=1,
            type="COMPOUND",
            settings={},
            connections=[", ".join(i) for i in connections],
            ports={k: ", ".join(v) for k, v in ports_map.items()},
        )
    )
    c.netlist = netlist
    return c


if __name__ == "__main__":
    import pp
    from pp.components.ring_single_bus import ring_single_bus_netlist

    components, connections, ports_map = ring_single_bus_netlist()
    c = netlist_to_component(components, connections, ports_map)
    print(c.netlist)
    pp.show(c)

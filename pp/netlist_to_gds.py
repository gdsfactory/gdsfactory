""" Deprecated, use componet_from_yaml instead

"""

from typing import Dict, List, Tuple

import numpy as np

from pp.component import Component, ComponentReference
from pp.types import Coordinate

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


def gen_sref(
    component: Component,
    transformation_name: str,
    port_name: str,
    position: Coordinate,
) -> ComponentReference:
    """Returns a Reference."""

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

    ref = ComponentReference(component=component, origin=(0, 0))

    if x_reflection:  # Vertical mirror: Reflection across x-axis
        y0 = port_position[1]
        ref.reflect(p1=(0, y0), p2=(1, y0))

    if rotation_angle != 0:
        ref.rotate(rotation_angle, center=port_position)

    translation = np.array(position - port_position)

    ref.move(destination=translation)

    return ref


def netlist_to_component(
    instances: Dict[str, Tuple[Component, str]],
    connections: List[Tuple[str, str, str, str]],
    ports_map: Dict[str, Tuple[str, str]] = None,
    position: Coordinate = (0.0, 0.0),
) -> Component:
    """Netlist_to_component is deprecated! use pp.componet_from_yaml instead
    Returns a component from a netlist (instances, connections and ports map)

    Args:
        instances:
            list of (instance_id <str>, component <Component>, transform <tuple>)

        connections:
            list of (component_id1, port_name1, component_id2, port_name2)
            Has to be ordered such that any component refered to in the connections
            has already been placed (except for the first component)

        ports_map:
            {port_name: (instance_id, port_name)}

    Returns: component with netlist stored in component.netlist

    ```

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
            "instances": [
                "CP1, CP2x2, None, None, 0.0, 0.0",
                "WG1, WG_CAVITY, None, None, 0.0, 0.0",
            ],
            "connections": ["CP1, out1, WG1, in", "WG1, out, CP1, in1"],
            "ports": {"in1": "CP1, in2", "out1": "CP1, out2"},
        },
    ]
    ```

    mirror, rotation, x, y
    """

    if len(connections) == 0:
        raise ValueError("no connections defined")

    instance_id, port, _, _ = connections[0]
    assert instance_id in instances, f"{instance_id} not in {list(instances.keys())}"
    component, transform_name = instances[instance_id]

    # First component reference
    sref_start = gen_sref(component, transform_name, port, position)
    cmp_name_to_sref = {instance_id: sref_start}

    # Iterate over all connections: create and place instances
    for cmp1_name, port1, cmp2_name, port2 in connections:
        if cmp1_name not in cmp_name_to_sref:
            cmp1_name, port1, cmp2_name, port2 = cmp2_name, port2, cmp1_name, port1

        if cmp2_name not in cmp_name_to_sref:
            component, transform_name = instances[cmp2_name]

            _ref = cmp_name_to_sref[cmp1_name]
            try:
                position = _ref.ports[port1].midpoint
            except Exception:
                print("{} has not port {}".format(cmp1_name, port1))

            sref = gen_sref(component, transform_name, port2, position)

            cmp_name_to_sref[cmp2_name] = sref

    c = Component()
    c.add(list(cmp_name_to_sref.values()))

    if ports_map:
        for port_name, (cmp_id, internal_port_name) in ports_map.items():
            component_ref = cmp_name_to_sref[cmp_id]
            component = component_ref.parent
            assert internal_port_name in component.ports, (
                f"{internal_port_name} not in {component_ref.ports.keys()} for"
                f" {component}"
            )
            port = component_ref.ports[internal_port_name]
            c.add_port(port_name, port=port)
            ports_map = {k: ", ".join(v) for k, v in ports_map.items()}

    # Set aliases
    for cmp_id, ref in cmp_name_to_sref.items():
        c[cmp_id] = ref

    c.netlist = cmp_name_to_sref

    # add leaf cells to netlist
    netlist = []
    for name, (component, _) in instances.items():
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
            ports=ports_map,
        )
    )
    c.netlist = netlist
    return c

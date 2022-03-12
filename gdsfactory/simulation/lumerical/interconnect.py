from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
from omegaconf import DictConfig

c = 2.9979e8
pi = np.pi


def add_interconnect_element(
    inc: object,
    label: str,
    model: str,
    loc: Tuple[int, int] = (200.0, 200.0),
    flip_vert: bool = False,
    flip_horiz: bool = False,
    rotation: float = 0.0
    # **kwargs
):
    """

    Args:
        inc: Interconnect instance
        label: label for Interconnect component
        model:
        loc:
        flip_vert:
        flip_horiz:
        rotation:

    """
    props = OrderedDict(
        [
            ("name", label),
            ("x position", loc[0]),
            ("y position", loc[1]),
            ("horizontal flipped", float(flip_horiz)),
            ("vertical flipped", float(flip_vert)),
            ("rotated", rotation)
            # kwargs
        ]
    )
    return inc.addelement(model, properties=props)


def send_to_interconnect(
    component,
    session: Optional[object] = None,
    placements: dict = None,
    simulation_settings: OrderedDict = None,
    run: bool = True,
    drop_port_prefix: str = None,
    **settings
) -> None:
    """Send component netlist to lumerical interconnect.

    Args:
        component: component from which to extract netlist
        session: Interconnect session
        placements: x,y pairs for where to place the components in the Interconnect GUI
        simulation_settings: global settings for Interconnect simulation
        run: whether to run or return the Interconnect sesson
        drop_port_prefix: if components are written with some prefix, drop up to and including
            the prefix character.  (i.e. "c1_input" -> "input")
    """
    import lumapi

    inc = session or lumapi.INTERCONNECT(hide=False)

    inc.switchtolayout()
    inc.deleteall()

    c = component

    netlist = c.get_netlist()

    instances: DictConfig = netlist["instances"]
    connections: DictConfig = netlist["connections"]
    placements: DictConfig = netlist["placements"] if not placements else placements

    for i, instance in enumerate(instances):
        model = instances[instance].model
        # component_settings = instances[instance].settings

        add_interconnect_element(
            inc=inc,
            label=instance,
            model=model,
            loc=(placements[instance].x, placements[instance].y),
            rotation=placements[instance].rotation,
            # **component_settings
        )

    for connection in connections:
        element2, port2 = connection.split(",")
        element1, port1 = connections[connection].split(",")

        if drop_port_prefix:
            port1 = port1[port1.index(drop_port_prefix) + 1 :]
            port2 = port2[port2.index(drop_port_prefix) + 1 :]

        # EBeam ports are not named consistently between Klayout and Interconnect..
        # Best to use another
        if hasattr(instances[element1]["info"], port1):
            port1 = instances[element1]["info"][port1]
        if hasattr(instances[element2]["info"], port2):
            port2 = instances[element2]["info"][port2]
        inc.connect(element1, port1, element2, port2)

    if simulation_settings:
        for param, val in zip(simulation_settings.keys(), simulation_settings.values()):
            inc.setnamed("::Root Element", param, val)

    if run:
        # There's nothing to run yet so this will give an error
        inc.run()
    else:
        return inc


if __name__ == "__main__":
    import ubcpdk.components as pdk

    import gdsfactory as gf

    c = gf.Component()
    gc1 = c << pdk.gc_te1550()
    gc2 = c << pdk.gc_te1550()
    gc3 = c << pdk.gc_te1550()

    s = c << pdk.y_splitter()

    gc1.connect(port="opt1", destination=s.ports["opt1"])
    gc2.connect(port="opt1", destination=s.ports["opt2"])
    gc3.connect(port="opt1", destination=s.ports["opt3"])

    c.show()

    simulation_settings = OrderedDict(
        [
            ("bitrate", 2.5e10),
        ]
    )

    send_to_interconnect(c, simulation_settings=simulation_settings, run=False)

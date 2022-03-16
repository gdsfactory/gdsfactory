from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from gdsfactory import Component

c = 2.9979e8
pi = np.pi
um = 1e-6


def set_global_settings(session: object, simulation_settings: dict):
    for param, val in zip(simulation_settings.keys(), simulation_settings.values()):
        session.setnamed("::Root Element", param, val)


def add_interconnect_element(
    session: object,
    label: str,
    model: str,
    loc: Tuple[int, int] = (200.0, 200.0),
    flip_vert: bool = False,
    flip_horiz: bool = False,
    rotation: float = 0.0,
    extra_props: OrderedDict = None,
):
    """
    Add an element to the Interconnect session.
    TODO: Need to connect this to generated s-parameters and add them to the model as well

    Args:
        session: Interconnect session
        label: label for Interconnect component
        model:
        loc:
        flip_vert:
        flip_horiz:
        rotation:
        extra_props:

    """
    props = OrderedDict(
        [
            ("name", label),
            ("x position", loc[0]),
            ("y position", loc[1]),
            ("horizontal flipped", float(flip_horiz)),
            ("vertical flipped", float(flip_vert)),
            ("rotated", rotation),
        ]
    )
    if extra_props:
        props.update(extra_props)
    return session.addelement(model, properties=props)


def send_to_interconnect(
    component: Component,
    session: Optional[object] = None,
    placements: dict = None,
    simulation_settings: OrderedDict = None,
    drop_port_prefix: str = None,
    component_distance_scaling: float = 1,
    **settings,
) -> object:
    """Send all components in netlist to Interconnect and make connections according to netlist.

    Args:
        component: component from which to extract netlist
        session: Interconnect session
        placements: x,y pairs for where to place the components in the Interconnect GUI
        simulation_settings: global settings for Interconnect simulation
        drop_port_prefix: if components are written with some prefix, drop up to and including
            the prefix character.  (i.e. "c1_input" -> "input")
        component_distance_scaling: scaling factor for component distances when laying out Interconnect schematic
    """
    import sys

    if "lumapi" not in sys.modules.keys():
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
        info = instances[instance].info
        extra_props = info["interconnect"] if "interconnect" in info.keys() else None
        add_interconnect_element(
            session=inc,
            label=instance,
            model=info.model,
            loc=(
                component_distance_scaling * placements[instance].x,
                component_distance_scaling * placements[instance].y,
            ),
            rotation=placements[instance].rotation,
            extra_props=extra_props,
        )

    for connection in connections:
        element2, port2 = connection.split(",")
        element1, port1 = connections[connection].split(",")

        if drop_port_prefix:
            # a bad way to autodetect which ports need to have prefixes dropped..
            try:
                port1 = port1[port1.index(drop_port_prefix) + 1 :]
            except:
                pass
            try:
                port2 = port2[port2.index(drop_port_prefix) + 1 :]
            except:
                pass

        # EBeam ports are not named consistently between Klayout and Interconnect..
        if hasattr(instances[element1].info, port1):
            port1 = instances[element1].info[port1]
        if hasattr(instances[element2].info, port2):
            port2 = instances[element2].info[port2]

        inc.connect(element1, port1, element2, port2)

    if simulation_settings:
        set_global_settings(inc, simulation_settings)
    return inc


def run_wavelength_sweep(
    component: Component,
    session: Optional[object] = None,
    port_in: Tuple = None,
    ports_out: Tuple = None,
    wavelength_range: Tuple = (1.500, 1.600),
    n_points: int = 1000,
    results: Tuple = ("transmission",),
    extra_ona_props: dict = None,
    **kwargs,
) -> dict:
    """
    Args:
        component:
        session:
        port_in: specify the port in the Interconnect model to attach the ONA output to
        ports_out: specify the ports in the Interconnect models to attach the ONA input to
        wavelength_range:
        n_points:
        results:
        extra_ona_props:
        kwargs:

    """
    import lumapi

    inc = session if session else lumapi.INTERCONNECT()

    # Add Monte-Carlo params
    inc.addproperty("::Root Element", "MC_uniformity_thickness", "wafer", "Matrix")
    inc.addproperty("::Root Element", "MC_uniformity_width", "wafer", "Matrix")
    inc.addproperty("::Root Element", "MC_non_uniform", "wafer", "Number")
    inc.addproperty("::Root Element", "MC_grid", "wafer", "Number")
    inc.addproperty("::Root Element", "MC_resolution_x", "wafer", "Number")
    inc.addproperty("::Root Element", "MC_resolution_y", "wafer", "Number")

    # send circuit to interconnect
    inc = send_to_interconnect(component=component, session=inc, **kwargs)

    ona_props = OrderedDict(
        [
            ("number of input ports", len(ports_out)),
            ("number of points", n_points),
            ("input parameter", "start and stop"),
            ("start frequency", (c / (wavelength_range[1] * um))),
            ("stop frequency", (c / (wavelength_range[0] * um))),
            ("plot kind", "wavelength"),
            ("relative to center", float(False)),
        ]
    )
    if extra_ona_props:
        ona_props.update(extra_ona_props)

    ona = add_interconnect_element(
        session=inc,
        model="Optical Network Analyzer",
        label="ONA_1",
        loc=(0, -50),
        extra_props=ona_props,
    )

    inc.connect(ona.name, "output", *port_in)
    for i, port in enumerate(ports_out):
        inc.connect(ona.name, f"input {i+1}", *ports_out[i])

    inc.run()

    data = dict()
    for result in results:
        data[result] = {
            port: inc.getresult(ona.name, f"input {i+1}/mode 1/{result}")
            for i, port in enumerate(ports_out)
        }

    # inc.close()
    return data


if __name__ == "__main__":
    import ubcpdk.components as pdk

    import gdsfactory as gf

    component = gf.Component()
    gc1 = component << pdk.gc_te1550()
    gc2 = component << pdk.gc_te1550()

    s = component << pdk.y_splitter()

    gc1.connect(port="opt1", destination=s.ports["opt1"])
    gc2.connect(port="opt1", destination=s.ports["opt2"])

    component.show()

    netlist = component.get_netlist()

    simulation_settings = OrderedDict(
        [
            ("MC_uniformity_thickness", np.array([200, 200])),
            ("MC_uniformity_width", np.array([200, 200])),
            ("MC_non_uniform", 0),
            ("MC_grid", 1e-5),
            ("MC_resolution_x", 200),
            ("MC_resolution_y", 0),
        ]
    )

    # inc = send_to_interconnect(
    #     component=c,
    #     simulation_settings=simulation_settings
    # )
    from gdsfactory.get_netlist import get_instance_name

    gc1_netlist_instance_name = get_instance_name(component, gc1)
    gc2_netlist_instance_name = get_instance_name(component, gc2)

    ports_out = (
        (gc2_netlist_instance_name, "opt_fiber"),
        (gc1_netlist_instance_name, "opt_wg"),
    )
    results = run_wavelength_sweep(
        component=component,
        port_in=(gc1_netlist_instance_name, "opt_fiber"),
        ports_out=ports_out,
        simulation_settings=simulation_settings,
        results=("transmission",),
    )

    import matplotlib.pyplot as plt

    plt.figure()
    for port in ports_out:
        wl = results["transmission"][port]["wavelength"] / um
        T = 10 * np.log10(np.abs(results["transmission"][port]["TE transmission"]))
        plt.plot(wl, T, label=f"{port}")
    plt.legend()
    plt.xlabel(r"Wavelength ($\mu$m)")
    plt.ylabel("TE transmission (dB)")
    plt.show()
    pass

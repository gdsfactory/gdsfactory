from __future__ import annotations

import contextlib
import pathlib
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from gdsfactory import Component
from gdsfactory.config import PATH

c = 2.9979e8
pi = np.pi
um = 1e-6


def install_design_kit(
    session: object,
    install_dir: pathlib.Path = PATH.interconnect,
    overwrite: bool = False,
) -> None:
    from gdsfactory.pdk import get_interconnect_cml_path

    cml_path = get_interconnect_cml_path()
    session.installdesignkit(str(cml_path), str(install_dir), overwrite)


def set_named_settings(
    session: object, simulation_settings: dict, element: str
) -> None:
    for param, val in zip(simulation_settings.keys(), simulation_settings.values()):
        session.setnamed(element, param, val)


# TODO: Add custom s-parameter models using compound elements


def add_interconnect_element(
    session: object,
    label: str,
    model: str,
    loc: Tuple[float, float] = (200.0, 200.0),
    flip_vert: bool = False,
    flip_horiz: bool = False,
    rotation: float = 0.0,
    simulation_props: Optional[OrderedDict] = None,
):
    """Add an element to the Interconnect session.

    TODO: Need to connect this to generated s-parameters and add them to the model as well

    Args:
        session: Interconnect session.
        label: label for Interconnect component.
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
    if simulation_props:
        if "library" in simulation_props.keys():
            _ = simulation_props.pop("library")
        if "properties" in simulation_props.keys():
            props.update(simulation_props["properties"])
        else:
            props.update(simulation_props)
    return session.addelement(model, properties=props)


def get_interconnect_settings(instance):
    info = instance.info.copy()
    if "interconnect" not in info.keys():
        return {}
    settings = info["interconnect"]
    if "properties" not in settings:
        settings["properties"] = []
    if "layout_model_property_pairs" in settings.keys():
        pairs = settings.pop("layout_model_property_pairs")
        for inc_name, (layout_name, scale) in pairs.items():
            settings["properties"][inc_name] = info[layout_name] * scale
    return settings


def send_to_interconnect(
    component: Component,
    session: object,
    ports_in: Optional[dict] = None,
    ports_out: Optional[dict] = None,
    placements: Optional[dict] = None,
    simulation_settings: Optional[OrderedDict] = None,
    drop_port_prefix: Optional[str] = None,
    component_distance_scaling_x: float = 1,
    component_distance_scaling_y: float = 1,
    setup_mc: bool = False,
    exclude_electrical: bool = True,
    **settings,
) -> object:
    """Send netlist components to Interconnect and connect them according to netlist.

    Args:
        component: component from which to extract netlist.
        session: Interconnect session.
        placements: x,y pairs for where to place the components in the Interconnect GUI.
        simulation_settings: global settings for Interconnect simulation.
        drop_port_prefix: if components are written with some prefix, drop up to and including
            the prefix character.  (i.e. "c1_input" -> "input").
        component_distance_scaling: scaling factor for component distances when
            laying out Interconnect schematic.

    """
    if not session:
        import lumapi

        session = lumapi.INTERCONNECT()

    install_design_kit(session=session)

    session.switchtolayout()
    session.deleteall()

    # Make compound element for circuit
    compound_element = session.addelement(
        "Compound Element", properties=OrderedDict([("name", str(component.name))])
    )
    if setup_mc:
        MC_param_element = f"::Root Element::{component.name}"
        # Add Monte-Carlo params
        session.addproperty(
            MC_param_element, "MC_uniformity_thickness", "wafer", "Matrix"
        )
        session.addproperty(MC_param_element, "MC_uniformity_width", "wafer", "Matrix")
        session.addproperty(MC_param_element, "MC_non_uniform", "wafer", "Number")
        session.addproperty(MC_param_element, "MC_grid", "wafer", "Number")
        session.addproperty(MC_param_element, "MC_resolution_x", "wafer", "Number")
        session.addproperty(MC_param_element, "MC_resolution_y", "wafer", "Number")

    # Switch groupscope to compound element so that anything added will go into it
    session.groupscope(compound_element.name)

    c = component

    netlist = c.get_netlist()

    instances: DictConfig = netlist["instances"]
    connections: DictConfig = netlist["connections"]
    placements: DictConfig = placements or netlist["placements"]
    ports: DictConfig = netlist["ports"]

    relay_count = 1
    excluded = []
    for instance in instances:
        if exclude_electrical:
            # Exclude if purely electrical
            with contextlib.suppress(Exception):
                port_types = instances[instance].full["cross_section"].port_types
                if ("electrical" in port_types) and ("optical" not in port_types):
                    excluded.append(instance)
                    continue

        loc = (
            component_distance_scaling_x * placements[instance].x,
            component_distance_scaling_y * placements[instance].y,
        )

        sim_props = get_interconnect_settings(instances[instance])

        if "model" not in sim_props.keys():
            raise KeyError(f"Please specify an interconnect model for {instance!r}")
        model = sim_props.pop("model")

        if "layout_model_port_pairs" in sim_props.keys():
            _ = sim_props.pop("layout_model_port_pairs")

        add_interconnect_element(
            session=session,
            label=instance,
            loc=loc,
            rotation=float(placements[instance].rotation),
            model=model,
            simulation_props=sim_props,
        )
        if instance in ports_in:
            # Add input port and connect to the compound element input port
            session.addport(
                f"::Root Element::{compound_element.name}",
                f"{instance}.{ports_in[instance]}",
                "input",
                "Optical Signal",
                "left",
            )
            session.connect(
                f"::Root Element::{compound_element.name}::RELAY_{relay_count}",
                "output",
                instance,
                ports_in[instance],
            )
            session.setnamed(
                f"::Root Element::{compound_element.name}::RELAY_{relay_count}",
                "y position",
                loc[1] - 50,
            )
            session.setnamed(
                f"::Root Element::{compound_element.name}::RELAY_{relay_count}",
                "x position",
                loc[0] - 50,
            )
            relay_count += 1
        elif instance in ports_out:
            session.addport(
                f"::Root Element::{compound_element.name}",
                f"{instance}.{ports_out[instance]}",
                "output",
                "Optical Signal",
                "right",
            )
            session.connect(
                instance,
                ports_out[instance],
                f"::Root Element::{compound_element.name}::RELAY_{relay_count}",
                "input",
            )
            session.setnamed(
                f"::Root Element::{compound_element.name}::RELAY_{relay_count}",
                "y position",
                loc[1] + 50,
            )
            session.setnamed(
                f"::Root Element::{compound_element.name}::RELAY_{relay_count}",
                "x position",
                loc[0] + 50,
            )
            relay_count += 1

    if ports:
        for port in ports_in:
            input_instance, instance_port = ports[port].split(",")
            info = get_interconnect_settings(instances[input_instance])
            if "layout_model_port_pairs" in info.keys():
                instance_port = info["layout_model_port_pairs"][instance_port]
            session.addport(
                f"::Root Element::{compound_element.name}",
                str(port),
                "input",
                "Optical Signal",
                "left",
            )
            session.connect(
                f"::Root Element::{compound_element.name}::RELAY_{relay_count}",
                "output",
                input_instance,
                instance_port,
            )
            session.setnamed(
                f"::Root Element::{compound_element.name}::RELAY_{relay_count}",
                "y position",
                session.getnamed(input_instance, "y position") - 50,
            )
            session.setnamed(
                f"::Root Element::{compound_element.name}::RELAY_{relay_count}",
                "x position",
                session.getnamed(input_instance, "x position") - 50,
            )
            relay_count += 1
        for port in ports_out:
            output_instance, instance_port = ports[port].split(",")
            info = get_interconnect_settings(instances[output_instance])
            if "layout_model_port_pairs" in info.keys():
                instance_port = info["layout_model_port_pairs"][instance_port]
            session.addport(
                f"::Root Element::{compound_element.name}",
                str(port),
                "output",
                "Optical Signal",
                "right",
            )
            session.connect(
                output_instance,
                instance_port,
                f"::Root Element::{compound_element.name}::RELAY_{relay_count}",
                "input",
            )
            session.setnamed(
                f"::Root Element::{compound_element.name}::RELAY_{relay_count}",
                "y position",
                session.getnamed(output_instance, "y position") + 50,
            )
            session.setnamed(
                f"::Root Element::{compound_element.name}::RELAY_{relay_count}",
                "x position",
                session.getnamed(output_instance, "x position") + 50,
            )
            relay_count += 1

    for connection in connections:
        element2, port2 = connection.split(",")
        element1, port1 = connections[connection].split(",")
        if (element1 in excluded) or (element2 in excluded):
            continue
        if drop_port_prefix:
            # a bad way to autodetect which ports need to have prefixes dropped.
            with contextlib.suppress(Exception):
                port1 = port1[port1.index(drop_port_prefix) + 1 :]
            with contextlib.suppress(Exception):
                port2 = port2[port2.index(drop_port_prefix) + 1 :]

        # EBeam ports are not named consistently between Klayout and Interconnect..
        element1_info = get_interconnect_settings(instances[element1])
        if "layout_model_port_pairs" in element1_info.keys():
            port1 = element1_info["layout_model_port_pairs"][port1]
        element2_info = get_interconnect_settings(instances[element2])
        if "layout_model_port_pairs" in element2_info.keys():
            port2 = element2_info["layout_model_port_pairs"][port2]

        session.connect(element1, port1, element2, port2)

    if simulation_settings:
        set_named_settings(
            session, simulation_settings, element=f"::Root Element::{component.name}"
        )
    # Return to highest-level element
    session.groupscope("::Root Element")

    # Auto-distribute ports on the compound element
    session.autoarrange(component.name)
    return session


def run_wavelength_sweep(
    component: Component,
    session: Optional[object] = None,
    setup_simulation: bool = True,
    is_top_level: bool = False,
    ports_in: Optional[dict] = None,
    ports_out: Optional[dict] = None,
    mode: int = 1,
    wavelength_range: Tuple[float, float] = (1.500, 1.600),
    n_points: int = 1000,
    results: Tuple[str, ...] = ("transmission",),
    extra_ona_props: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Args are the following.

    component:
    session:
    setup_simulation: whether to send the component to interconnect before running the sweep.
    ports_in: specify the port in the Interconnect model to attach the ONA output to.
    ports_out: specify the ports in the Interconnect models to attach the ONA input to.
    wavelength_range:
    n_points:
    results:
    extra_ona_props:
    kwargs:
    """
    if len(ports_in) > 1:
        raise ValueError("Only 1 input port is supported at this time")

    import lumapi

    if not session:
        session = lumapi.INTERCONNECT()

    install_design_kit(session=session)

    if setup_simulation:
        session = send_to_interconnect(
            component=component,
            session=session,
            ports_in=ports_in,
            ports_out=ports_out,
            **kwargs,
        )

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
        session=session,
        model="Optical Network Analyzer",
        label="ONA_1",
        loc=(0, -50),
        simulation_props=ona_props,
    )
    for port in ports_in.keys():
        name = port if is_top_level else f"{port}.{ports_in[port]}"
        session.connect(ona.name, "output", component.name, name)
    for i, port in enumerate(ports_out.keys()):
        name = port if is_top_level else f"{port}.{ports_out[port]}"
        session.connect(ona.name, f"input {i+1}", component.name, name)

    session.run()

    # inc.close()
    return {
        result: {
            port: session.getresult(ona.name, f"input {i+1}/mode {mode}/{result}")
            for i, port in enumerate(ports_out)
        }
        for result in results
    }


def plot_wavelength_sweep(
    ports_out, results, result_name: str = "TE Transmission", show: bool = True
) -> None:
    import matplotlib.pyplot as plt

    for port in ports_out:
        wl = results["transmission"][port]["wavelength"] / um
        T = 10 * np.log10(np.abs(results["transmission"][port][result_name]))
        plt.plot(wl, T, label=str(port))
    plt.legend()
    plt.xlabel(r"Wavelength ($\mu$m)")
    plt.ylabel(f"{result_name} (dB)")

    if show:
        plt.show()


if __name__ == "__main__":
    import ubcpdk.components as pdk

    mzi = pdk.mzi()

    netlist = mzi.get_netlist()

    ports_in = {"o1": "o1"}
    ports_out = {"o2": "o2"}

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
    results = run_wavelength_sweep(
        component=mzi,
        ports_in=ports_in,
        ports_out=ports_out,
        results=("transmission",),
        component_distance_scaling=50,
        simulation_settings=simulation_settings,
        setup_mc=True,
        is_top_level=True,
    )

    plot_wavelength_sweep(ports_out=ports_out, results=results)

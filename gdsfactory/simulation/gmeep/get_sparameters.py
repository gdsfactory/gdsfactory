"""Compute and write Sparameters using Meep."""

import pathlib
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd
import pydantic

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.config import CONFIG
from gdsfactory.simulation.get_sparameters_path import get_sparameters_path
from gdsfactory.simulation.gmeep.get_simulation import (
    LAYER_TO_MATERIAL,
    LAYER_TO_THICKNESS,
    get_simulation,
)


@pydantic.validate_arguments
def get_sparameters1x1(
    component: Component,
    dirpath: Path = CONFIG["sparameters"],
    layer_to_thickness: Dict[Tuple[int, int], float] = LAYER_TO_THICKNESS,
    layer_to_material: Dict[Tuple[int, int], str] = LAYER_TO_MATERIAL,
    filepath: Optional[Path] = None,
    overwrite: bool = False,
    **settings,
) -> pd.DataFrame:
    """Compute Sparameters and writes them in CSV filepath.

    Args:
        component: to simulate.
        dirpath: directory to store Sparameters
        layer_to_thickness: GDS layer (int, int) to thickness
        layer_to_material: GDS layer (int, int) to material string ('Si', 'SiO2', ...)
        filepath: to store pandas Dataframe with Sparameters in CSV format
        overwrite: overwrites
        **settings: sim settings

    Returns:
        sparameters in a pandas Dataframe

    """
    filepath = filepath or get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_to_material=layer_to_material,
        layer_to_thickness=layer_to_thickness,
    )
    filepath = pathlib.Path(filepath)
    if filepath.exists() and not overwrite:
        print(f"Simulation loaded from file: {filepath}")
        return pd.read_csv(filepath)

    sim_dict = get_simulation(
        component=component,
        layer_to_thickness=layer_to_thickness,
        layer_to_material=layer_to_material,
        **settings,
    )

    sim = sim_dict["sim"]
    monitors = sim_dict["monitors"]
    freqs = sim_dict["freqs"]
    field_monitor_point = sim_dict["field_monitor_point"]
    wavelengths = 1 / freqs
    port1 = sim_dict["port_source_name"]
    port2 = set(monitors.keys()) - set([port1])
    port2 = list(port2)[0]
    monitor1 = monitors[port1]
    monitor2 = monitors[port2]

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            dt=50, c=mp.Ez, pt=field_monitor_point, decay_by=1e-9
        )
    )
    # call this function every 50 time spes
    # look at simulation and measure component that we want to measure (Ez component)
    # when field_monitor_point decays below a certain 1e-9 field threshold

    # Calculate mode overlaps
    m_results = np.abs(sim.get_eigenmode_coefficients(monitor1, [1]).alpha)
    a = m_results[:, :, 0]  # forward wave
    source_fields = np.squeeze(a)
    t = sim.get_eigenmode_coefficients(monitor2, [1]).alpha[0, :, 0] / source_fields
    r = sim.get_eigenmode_coefficients(monitor1, [1]).alpha[0, :, 1] / source_fields

    s11 = r
    s12 = t

    s11a = np.unwrap(np.angle(s11))
    s12a = np.unwrap(np.angle(s12))
    s11m = np.abs(s11)
    s12m = np.abs(s12)

    df = pd.DataFrame(
        dict(
            wavelengths=wavelengths,
            freqs=freqs,
            s11m=s11m,
            s11a=s11a,
            s12m=s12m,
            s12a=s12a,
            s22m=s11m,
            s22a=s11a,
            s21m=s12m,
            s21a=s12a,
        )
    )
    print(f"transmission: {t}")

    return df


@pydantic.validate_arguments
def get_sparametersNxN(
    component: Component,
    wl_min: float = 1.5,
    wl_max: float = 1.6,
    wl_steps: int = 50,
    dirpath: Path = CONFIG["sparameters"],
    layer_to_thickness: Dict[Tuple[int, int], float] = LAYER_TO_THICKNESS,
    layer_to_material: Dict[Tuple[int, int], str] = LAYER_TO_MATERIAL,
    filepath: Optional[Path] = None,
    overwrite: bool = False,
    **settings,
) -> pd.DataFrame:
    """Compute Sparameters and writes them in CSV filepath.
    Repeats the simulation, each time using a different port in (by default, all of them)
    TODO : user can provide list of port name tuples in symmetries

    Args:
        component: to simulate.
        source_ports: list of port string names to use as sources
        dirpath: directory to store Sparameters
        layer_to_thickness: GDS layer (int, int) to thickness
        layer_to_material: GDS layer (int, int) to material string ('Si', 'SiO2', ...)
        filepath: to store pandas Dataframe with Sparameters in CSV format
        overwrite: overwrites
        **settings: sim settings

    Returns:
        sparameters in a pandas Dataframe

    """
    filepath = filepath or get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_to_material=layer_to_material,
        layer_to_thickness=layer_to_thickness,
    )
    filepath = pathlib.Path(filepath)
    if filepath.exists() and not overwrite:
        print(f"Simulation loaded from file: {filepath}")
        return pd.read_csv(filepath)

    # Parse ports
    Sparams_indices = []
    component_ref = component.ref()
    for port_name in component_ref.ports.keys():
        if component_ref.ports[port_name].port_type == "optical":
            Sparams_indices.append(re.findall("[0-9]+", port_name)[0])

    # Create S-parameter storage object
    # Sparams_array = np.zeros([2*len(Sparams_indices), 2*len(Sparams_indices), wl_steps])
    Sparams_dict = {}

    # Since source is defined upon sim object instanciation, loop here
    for port_index in Sparams_indices:

        sim_dict = get_simulation(
            component=component,
            port_source_name="o{}".format(port_index),
            wl_min=wl_min,
            wl_max=wl_max,
            wl_steps=wl_steps,
            port_margin=2,
            res=20,
            **settings,
        )

        sim = sim_dict["sim"]
        monitors = sim_dict["monitors"]
        freqs = sim_dict["freqs"]
        # field_monitor_point = sim_dict["field_monitor_point"]
        wavelengths = 1 / freqs

        # Make termination when field decayed enough across ALL monitors
        termination = []
        for monitor_name in monitors:
            termination.append(
                mp.stop_when_fields_decayed(
                    dt=50,
                    c=mp.Ez,
                    pt=monitors[monitor_name].regions[0].center,
                    decay_by=1e-9,
                )
            )

        sim.run(until_after_sources=termination)
        # call this function every 50 time spes
        # look at simulation and measure component that we want to measure (Ez component)
        # when field_monitor_point decays below a certain 1e-9 field threshold

        # # Calculate mode overlaps
        # Get source
        m_results = np.abs(
            sim.get_eigenmode_coefficients(
                monitors["o{}".format(port_index)], [1]
            ).alpha
        )
        a = m_results[:, :, 0]  # forward wave
        source_fields = np.squeeze(a)
        # Get coefficients
        for monitor_index in Sparams_indices:
            if monitor_index == port_index:
                r = (
                    sim.get_eigenmode_coefficients(
                        monitors["o{}".format(monitor_index)], [1]
                    ).alpha[0, :, 1]
                    / source_fields
                )
                sii = r
                siia = np.unwrap(np.angle(sii))
                siim = np.abs(sii)
                Sparams_dict["s{}{}a".format(port_index, monitor_index)] = siia
                Sparams_dict["s{}{}m".format(port_index, monitor_index)] = siim
            else:
                t = (
                    sim.get_eigenmode_coefficients(
                        monitors["o{}".format(monitor_index)], [1]
                    ).alpha[0, :, 0]
                    / source_fields
                )
                sij = t
                sija = np.unwrap(np.angle(sij))
                sijm = np.abs(sij)
                Sparams_dict["s{}{}a".format(port_index, monitor_index)] = sija
                Sparams_dict["s{}{}m".format(port_index, monitor_index)] = sijm

    df = pd.DataFrame(Sparams_dict)
    df["wavelengths"] = wavelengths
    df["freqs"] = freqs

    return df


def plot_sparameters(df: pd.DataFrame, **settings) -> None:
    """Plot Sparameters from a Pandas DataFrame."""
    wavelengths = df["wavelengths"]
    for key in df.keys():
        if key.endswith("m"):
            plt.plot(wavelengths, df[key], "-o", label=key, **settings)
    plt.ylabel("Power (dB)")
    plt.xlabel(r"Wavelength ($\mu$m)")
    plt.legend()
    plt.grid(True)


if __name__ == "__main__":

    # c = gf.components.bend_circular(radius=2)
    # c = gf.add_padding(c, default=0, bottom=2, right=2, layers=[(100, 0)])

    # c = gf.components.mmi1x2()
    # c = gf.add_padding(c.copy(), default=0, bottom=2, top=2, layers=[(100, 0)])

    # c = gf.components.mmi1x2()
    # c = gf.add_padding(c.copy(), default=0, bottom=2, top=2, layers=[(100, 0)])

    c = gf.components.mmi2x2(
        width=0.5,
        width_taper=1.0,
        length_taper=3,
        length_mmi=5.5,
        width_mmi=6,
        gap_mmi=2,
    )
    c = gf.add_padding(c.copy(), default=0, bottom=2, top=2, layers=[(100, 0)])

    # c = gf.components.straight(length=2)
    # c = gf.add_padding(c, default=0, bottom=2, top=2, layers=[(100, 0)])

    # sim_dict = get_simulation(c, is_3d=False)
    # df = get_sparameters1x1(c, overwrite=True)
    # plot_sparameters(df)
    # plt.show()
    # print(df)

    df = get_sparametersNxN(c, overwrite=True)
    plot_sparameters(df)
    plt.show()

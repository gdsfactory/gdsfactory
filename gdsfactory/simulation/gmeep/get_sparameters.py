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
    wavelengths = 1 / freqs
    field_monitor_point = sim_dict["field_monitor_point"]
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


def parse_port_eigenmode_coeff(port_index, ports, sim_dict):
    """
    Given a port and eigenmode coefficient result, returns the coefficients relative to whether the wavevector is entering or exiting simulation

    Args:
        port_index: index of port
        ports: component_ref.ports
        sim_dict:
    """
    # Inputs
    sim = sim_dict["sim"]
    monitors = sim_dict["monitors"]

    # Direction of port (pointing away from the simulation)
    angle_rad = np.radians(ports["o{}".format(port_index)].orientation)
    kpoint = mp.Vector3(x=1).rotate(mp.Vector3(z=1), angle_rad)

    # Get port coeffs
    monitor_coeff = sim.get_eigenmode_coefficients(
        monitors["o{}".format(port_index)], [1], kpoint_func=lambda f, n: kpoint
    )

    # Get port logical orientation
    # kdom = monitor_coeff.kdom[0] # Pick one wavelength, assume behaviour similar across others

    # get_eigenmode_coeff.alpha[:,:,idx] with ind being the forward or backward wave according to cell coordinates.
    # Figure out if that is exiting the simulation or not depending on the port orientation (assuming it's near PMLs)
    if ports["o{}".format(port_index)].orientation == 0:  # east
        idx_in = 1
        idx_out = 0
    elif ports["o{}".format(port_index)].orientation == 90:  # north
        idx_in = 1
        idx_out = 0
    elif ports["o{}".format(port_index)].orientation == 180:  # west
        idx_in = 0
        idx_out = 1
    elif ports["o{}".format(port_index)].orientation == 270:  # south
        idx_in = 0
        idx_out = 1
    else:
        ValueError("Port orientation is not 0, 90, 180, or 270 degrees!")

    # # Adjust according to whatever the monitor decided was positive
    # idx_out = 1 - (kdom*kpoint > 0) # if true 1 - 1, outgoing wave is the forward (0) wave
    # idx_in = 1 - idx_out
    # print('monitor_n = ', port_index)
    # print('kangle = ', kpoint)
    # print('kdom = ', kdom)
    # print('kdom*kpoint', kdom*kpoint)
    # print('idx (outgoing wave) = ', idx_out)
    # print('idx (ingoing wave) = ', idx_in)

    coeff_in = monitor_coeff.alpha[
        0, :, idx_in
    ]  # ingoing (w.r.t. simulation cell) wave
    coeff_out = monitor_coeff.alpha[
        0, :, idx_out
    ]  # outgoing (w.r.t. simulation cell) wave

    return coeff_in, coeff_out


@pydantic.validate_arguments
def get_sparametersNxN(
    component: Component,
    res: int = 20,
    wl_min: float = 1.5,
    wl_max: float = 1.6,
    wl_steps: int = 50,
    dirpath: Path = CONFIG["sparameters"],
    layer_to_thickness: Dict[Tuple[int, int], float] = LAYER_TO_THICKNESS,
    layer_to_material: Dict[Tuple[int, int], str] = LAYER_TO_MATERIAL,
    filepath: Optional[Path] = None,
    overwrite: bool = False,
    animate: bool = False,
    lazy_parallelism: bool = False,
    **settings,
) -> pd.DataFrame:
    """Compute Sparameters and writes them in CSV filepath.
    Repeats the simulation, each time using a different port in (by default, all of them)
    TODO : user can provide list of port name tuples whose results to merge (e.g. symmetric ports)

    Args:
        component: to simulate.
        source_ports: list of port string names to use as sources
        dirpath: directory to store Sparameters
        layer_to_thickness: GDS layer (int, int) to thickness
        layer_to_material: GDS layer (int, int) to material string ('Si', 'SiO2', ...)
        filepath: to store pandas Dataframe with Sparameters in CSV format
        overwrite: overwrites
        animate: saves a MP4 images of the simulation for inspection, and also outputs during computation. The name of the file is the source index
        lazy_parallelism: toggles the flag "meep.divide_parallel_processes" to perform the simulations with different sources in parallel
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
    Sparams_dict = {}

    @pydantic.validate_arguments
    def sparameter_calculation(
        n,
        component: Component,
        wl_min: float = 1.5,
        wl_max: float = 1.6,
        wl_steps: int = 50,
        dirpath: Path = CONFIG["sparameters"],
        layer_to_thickness: Dict[Tuple[int, int], float] = LAYER_TO_THICKNESS,
        layer_to_material: Dict[Tuple[int, int], str] = LAYER_TO_MATERIAL,
        animate: bool = False,
        **settings,
    ) -> Dict:

        sim_dict = get_simulation(
            component=component,
            port_source_name="o{}".format(Sparams_indices[n]),
            wl_min=wl_min,
            wl_max=wl_max,
            wl_steps=wl_steps,
            port_margin=2,
            port_monitor_offset=-0.1,
            port_source_offset=-0.1,
            **settings,
        )

        sim = sim_dict["sim"]
        monitors = sim_dict["monitors"]
        # freqs = sim_dict["freqs"]
        # wavelengths = 1 / freqs

        print(sim.resolution)

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

        if animate:
            sim.use_output_directory()
            animate = mp.Animate2D(
                sim,
                fields=mp.Ez,
                realtime=True,
                field_parameters={
                    "alpha": 0.8,
                    "cmap": "RdBu",
                    "interpolation": "none",
                },
                eps_parameters={"contour": True},
                normalize=True,
            )
            sim.run(mp.at_every(1, animate), until_after_sources=termination)
            animate.to_mp4(30, Sparams_indices[n] + ".mp4")
        else:
            sim.run(until_after_sources=termination)
        # call this function every 50 time spes
        # look at simulation and measure component that we want to measure (Ez component)
        # when field_monitor_point decays below a certain 1e-9 field threshold

        # # Calculate mode overlaps
        # Get source monitor results
        source_entering, source_exiting = parse_port_eigenmode_coeff(
            Sparams_indices[n], component_ref.ports, sim_dict
        )
        # Get coefficients
        for monitor_index in Sparams_indices:
            if monitor_index == Sparams_indices[n]:
                sii = source_exiting / source_entering
                siia = np.unwrap(np.angle(sii))
                siim = np.abs(sii)
                # Sparams_dict["sourceExiting{}{}".format(Sparams_indices[n], monitor_index)] = source_exiting
                # Sparams_dict["sourceEntering{}{}".format(Sparams_indices[n], monitor_index)] = source_entering
                Sparams_dict["s{}{}a".format(Sparams_indices[n], monitor_index)] = siia
                Sparams_dict["s{}{}m".format(Sparams_indices[n], monitor_index)] = siim
            else:
                monitor_entering, monitor_exiting = parse_port_eigenmode_coeff(
                    monitor_index, component_ref.ports, sim_dict
                )
                sij = monitor_exiting / source_entering
                sija = np.unwrap(np.angle(sij))
                sijm = np.abs(sij)
                # Sparams_dict["monitorExiting{}{}".format(Sparams_indices[n], monitor_index)] = monitor_exiting
                # Sparams_dict["monitorEntering{}{}".format(Sparams_indices[n], monitor_index)] = monitor_entering
                Sparams_dict["s{}{}a".format(Sparams_indices[n], monitor_index)] = sija
                Sparams_dict["s{}{}m".format(Sparams_indices[n], monitor_index)] = sijm
                sij = monitor_entering / source_entering
                sija = np.unwrap(np.angle(sij))
                sijm = np.abs(sij)
                # Sparams_dict["s{}{}a_enter".format(Sparams_indices[n], monitor_index)] = sija
                # Sparams_dict["s{}{}m_enter".format(Sparams_indices[n], monitor_index)] = sijm

        return Sparams_dict

    # Since source is defined upon sim object instanciation, loop here
    # for port_index in Sparams_indices:

    if lazy_parallelism:
        from mpi4py import MPI

        n = mp.divide_parallel_processes(len(Sparams_indices))
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        Sparams_dict = sparameter_calculation(
            n,
            component=component,
            wl_min=wl_min,
            wl_max=wl_max,
            wl_steps=wl_steps,
            layer_to_thickness=layer_to_thickness,
            layer_to_material=layer_to_material,
            animate=animate,
            **settings,
        )
        """
        Synchronize dicts
        From https://stackoverflow.com/questions/66703153/updating-dictionary-values-in-mpi4py
        """
        if rank == 0:
            for i in range(1, size, 1):
                data = comm.recv(source=i, tag=11)
                Sparams_dict.update(data)

            df = pd.DataFrame(Sparams_dict)
            df["wavelengths"] = np.linspace(wl_min, wl_max, wl_steps)
            df["freqs"] = 1 / df["wavelengths"]
            df.to_csv(filepath, index=False)
            return df
        else:
            comm.send(Sparams_dict, dest=0, tag=11)

    else:
        for n in range(len(Sparams_indices)):
            Sparams_dict.update(
                sparameter_calculation(
                    n,
                    component=component,
                    wl_min=wl_min,
                    wl_max=wl_max,
                    wl_steps=wl_steps,
                    layer_to_thickness=layer_to_thickness,
                    layer_to_material=layer_to_material,
                    animate=animate,
                    **settings,
                )
            )
        df = pd.DataFrame(Sparams_dict)
        df["wavelengths"] = np.linspace(wl_min, wl_max, wl_steps)
        df["freqs"] = 1 / df["wavelengths"]
        df.to_csv(filepath, index=False)
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

    # c = gf.components.mmi1x2(
    #     width=0.5,
    #     width_taper=1.0,
    #     length_taper=3,
    #     length_mmi=5.5,
    #     width_mmi=6,
    #     gap_mmi=2,
    # )
    # c2 = gf.add_padding(c.copy(), default=0, bottom=2, top=2, layers=[(100, 0)])

    # c = gf.components.coupler_full(length=20, gap=0.2, dw=0)
    # c2 = gf.add_padding(c.copy(), default=0, bottom=2, top=2, layers=[(100, 0)])

    c = gf.components.crossing()

    # sim_dict = get_simulation(c, is_3d=False)
    # df = get_sparameters1x1(c, overwrite=True)
    # plot_sparameters(df)
    # plt.show()
    # print(df)

    # df = get_sparametersNxN(c, filepath='./df_lazy_consolidated.csv', overwrite=True, animate=False, lazy_parallelism=True)
    df = get_sparametersNxN(
        c,
        filepath="./df_lazy_consolidated.csv",
        overwrite=True,
        animate=True,
        lazy_parallelism=True,
        resolution=120,
    )
    # df.to_csv("df_lazy.csv", index=False)
    # plot_sparameters(df)
    # plt.show()

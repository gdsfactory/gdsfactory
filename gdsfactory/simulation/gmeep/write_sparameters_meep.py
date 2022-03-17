"""Compute and write Sparameters using Meep."""

import inspect
import multiprocessing
import pathlib
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import meep as mp
import numpy as np
import pandas as pd
import pydantic
from omegaconf import OmegaConf
from tqdm import tqdm

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.config import logger, sparameters_path
from gdsfactory.simulation import port_symmetries
from gdsfactory.simulation.get_sparameters_path import (
    get_sparameters_path_meep as get_sparameters_path,
)
from gdsfactory.simulation.gmeep.get_simulation import (
    get_simulation,
    settings_get_simulation,
)
from gdsfactory.tech import LAYER_STACK, LayerStack
from gdsfactory.types import ComponentOrFactory, PathType, PortSymmetries

ncores = multiprocessing.cpu_count()


def remove_simulation_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a copy of dict with only simulation settings
    removes all flags for the simulator itself
    """
    d = d.copy()
    d.pop("run", None)
    d.pop("lazy_parallelism", None)
    d.pop("overwrite", None)
    d.pop("animate", None)
    d.pop("wait_to_finish", None)
    d.pop("cores", None)
    d.pop("temp_dir", None)
    d.pop("temp_file_str", None)
    return d


def parse_port_eigenmode_coeff(port_index: int, ports, sim_dict: Dict):
    """Given a port and eigenmode coefficient result, returns the coefficients
    relative to whether the wavevector is entering or exiting simulation

    Args:
        port_index: index of port
        ports: component_ref.ports
        sim_dict:
    """
    if f"o{port_index}" not in ports:
        raise ValueError(
            f"port = 'o{port_index}' not in {list(ports.keys())}. "
            "You can rename ports with Component.auto_rename_ports()"
        )

    # Inputs
    sim = sim_dict["sim"]
    monitors = sim_dict["monitors"]

    # Direction of port (pointing away from the simulation)

    # angle_rad = np.radians(ports[f"o{port_index}"].orientation)
    # kpoint = mp.Vector3(x=1).rotate(mp.Vector3(z=1), angle_rad)

    # # Get port coeffs
    # monitor_coeff = sim.get_eigenmode_coefficients(
    #     monitors[f"o{port_index}"], [1], kpoint_func=lambda f, n: kpoint
    # )

    # Get port logical orientation
    # kdom = monitor_coeff.kdom[0] # Pick one wavelength, assume behaviour similar across others

    # get_eigenmode_coeff.alpha[:,:,idx] with ind being the forward or backward wave according to cell coordinates.
    # Figure out if that is exiting the simulation or not
    # depending on the port orientation (assuming it's near PMLs)
    if ports[f"o{port_index}"].orientation == 0:  # east
        kpoint = mp.Vector3(x=1)
        idx_in = 1
        idx_out = 0
    elif ports[f"o{port_index}"].orientation == 90:  # north
        kpoint = mp.Vector3(y=1)
        idx_in = 1
        idx_out = 0
    elif ports[f"o{port_index}"].orientation == 180:  # west
        kpoint = mp.Vector3(x=1)
        idx_in = 0
        idx_out = 1
    elif ports[f"o{port_index}"].orientation == 270:  # south
        kpoint = mp.Vector3(y=1)
        idx_in = 0
        idx_out = 1
    else:
        ValueError("Port orientation is not 0, 90, 180, or 270 degrees!")

    # Get port coeffs
    monitor_coeff = sim.get_eigenmode_coefficients(
        monitors[f"o{port_index}"], [1], kpoint_func=lambda f, n: kpoint
    )

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
def write_sparameters_meep(
    component: ComponentOrFactory,
    port_symmetries: Optional[PortSymmetries] = None,
    resolution: int = 30,
    wavelength_start: float = 1.5,
    wavelength_stop: float = 1.6,
    wavelength_points: int = 50,
    dirpath: PathType = sparameters_path,
    layer_stack: LayerStack = LAYER_STACK,
    port_margin: float = 2,
    port_monitor_offset: float = -0.1,
    port_source_offset: float = -0.1,
    filepath: Optional[Path] = None,
    overwrite: bool = False,
    animate: bool = False,
    lazy_parallelism: bool = False,
    run: bool = True,
    dispersive: bool = False,
    xmargin: float = 0,
    ymargin: float = 3,
    xmargin_left: float = 0,
    xmargin_right: float = 0,
    ymargin_top: float = 0,
    ymargin_bot: float = 0,
    **settings,
) -> pd.DataFrame:
    r"""Compute Sparameters and writes them to a CSV filepath.
    Simulates each time using a different input port (by default, all of them)
    unless you specify port_symmetries:

    port_symmetries = {"o1":
            {
                "s11": ["s22","s33","s44"],
                "s21": ["s21","s34","s43"],
                "s31": ["s13","s24","s42"],
                "s41": ["s14","s23","s32"],
            }
        }
    - Only simulations using the outer key port names will be run
    - The associated value is another dict whose keys are the S-parameters computed
        when this source is active
    - The values of this inner Dict are lists of s-parameters whose values are copied

    This allows you doing less simulations

    TODO: automate this for common component types
    (geometrical symmetries, reciprocal materials, etc.)

    TODO: enable other port naming conventions, such as (in0, in1, out0, out1)


    .. code::

         top view
              ________________________________
             |                               |
             | xmargin_left                  | port_extension
             |<--------->       port_margin ||<-->
          o2_|___________          _________||_o3
             |           \        /          |
             |            \      /           |
             |             ======            |
             |            /      \           |
          o1_|___________/        \__________|_o4
             |   |                 <-------->|
             |   |ymargin_bot   xmargin_right|
             |   |                           |
             |___|___________________________|

        side view
              ________________________________
             |                     |         |
             |                     |         |
             |                   zmargin_top |
             |xmargin_left         |         |
             |<---> _____         _|___      |
             |     |     |       |     |     |
             |     |     |       |     |     |
             |     |_____|       |_____|     |
             |       |                       |
             |       |                       |
             |       |zmargin_bot            |
             |       |                       |
             |_______|_______________________|



    Args:
        component: to simulate.
        resolution: in pixels/um (30: for coarse, 100: for fine)
        port_symmetries: Dict to specify port symmetries, to save number of simulations
        dirpath: directory to store Sparameters
        layer_stack: LayerStack class
        port_margin: margin on each side of the port
        port_monitor_offset: offset between monitor Component port and monitor MEEP port
        port_source_offset: offset between source Component port and source MEEP port
        filepath: to store pandas Dataframe with Sparameters in CSV format.
            Defaults to dirpath/component_.csv
        overwrite: overwrites stored Sparameter CSV results.
        animate: saves a MP4 images of the simulation for inspection, and also
            outputs during computation. The name of the file is the source index
        lazy_parallelism: toggles the flag "meep.divide_parallel_processes" to
            perform the simulations with different sources in parallel
        run: runs simulation, if False, only plots simulation
        dispersive: use dispersive models for materials (requires higher resolution)
        xmargin: left and right distance from component to PML.
        xmargin_left: west distance from component to PML.
        xmargin_right: east distance from component to PML.
        ymargin: top and bottom distance from component to PML.
        ymargin_top: north distance from component to PML.
        ymargin_bot: south distance from component to PML.

    keyword Args:
        extend_ports_length: to extend ports beyond the PML (um).
        zmargin_top: thickness for cladding above core (um).
        zmargin_bot: thickness for cladding below core (um)
        tpml: PML thickness (um).
        clad_material: material for cladding.
        is_3d: if True runs in 3D
        wavelength_start: wavelength min (um).
        wavelength_stop: wavelength max (um).
        wavelength_points: wavelength steps
        dfcen: delta frequency
        port_source_name: input port name
        port_field_monitor_name:
        port_margin: margin on each side of the port (um).
        distance_source_to_monitors: in (um).
        port_source_offset: offset between source Component port and source MEEP port
        port_monitor_offset: offset between monitor Component port and monitor MEEP port
        material_name_to_meep: dispersive materials have a wavelength
            dependent index. Maps layer_stack names with meep material database names.

    Returns:
        sparameters in a pandas Dataframe (wavelengths, s11a, s12m, ...)
            where `a` is the angle in radians and `m` the module

    """
    component = component() if callable(component) else component
    assert isinstance(component, Component)

    for setting in settings.keys():
        if setting not in settings_get_simulation:
            raise ValueError(f"{setting} not in {settings_get_simulation}")

    port_symmetries = port_symmetries or {}

    xmargin_left = xmargin_left or xmargin
    xmargin_right = xmargin_right or xmargin

    ymargin_top = ymargin_top or ymargin
    ymargin_bot = ymargin_bot or ymargin

    sim_settings = dict(
        resolution=resolution,
        port_symmetries=port_symmetries,
        wavelength_start=wavelength_start,
        wavelength_stop=wavelength_stop,
        wavelength_points=wavelength_points,
        port_margin=port_margin,
        port_monitor_offset=port_monitor_offset,
        port_source_offset=port_source_offset,
        dispersive=dispersive,
        ymargin_top=ymargin_top,
        ymargin_bot=ymargin_bot,
        xmargin_left=xmargin_left,
        xmargin_right=xmargin_right,
        **settings,
    )

    filepath = filepath or get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_stack=layer_stack,
        **sim_settings,
    )

    sim_settings = sim_settings.copy()
    sim_settings["layer_stack"] = layer_stack.to_dict()
    sim_settings["component"] = component.to_dict()
    filepath = pathlib.Path(filepath)
    filepath_sim_settings = filepath.with_suffix(".yml")

    # filepath_sim_settings.write_text(OmegaConf.to_yaml(sim_settings))
    # logger.info(f"Write simulation settings to {filepath_sim_settings!r}")
    # return filepath_sim_settings

    component = gf.add_padding_container(
        component,
        default=0,
        top=ymargin_top,
        bottom=ymargin_bot,
        left=xmargin_left,
        right=xmargin_right,
    )

    if not run:
        sim_dict = get_simulation(
            component=component,
            wavelength_start=wavelength_start,
            wavelength_stop=wavelength_stop,
            wavelength_points=wavelength_points,
            layer_stack=layer_stack,
            port_margin=port_margin,
            port_monitor_offset=port_monitor_offset,
            port_source_offset=port_source_offset,
            **settings,
        )
        sim_dict["sim"].plot2D(plot_eps_flag=True)
        return

    if filepath.exists() and not overwrite:
        logger.info(f"Simulation loaded from {filepath!r}")
        return pd.read_csv(filepath)
    elif filepath.exists() and overwrite:
        filepath.unlink()

    # Parse ports (default)
    monitor_indices = []
    source_indices = []
    component_ref = component.ref()
    for port_name in component_ref.ports.keys():
        if component_ref.ports[port_name].port_type == "optical":
            monitor_indices.append(re.findall("[0-9]+", port_name)[0])
    if bool(port_symmetries):  # user-specified
        for port_name in port_symmetries.keys():
            source_indices.append(re.findall("[0-9]+", port_name)[0])
    else:  # otherwise cycle through all
        source_indices = monitor_indices

    # Create S-parameter storage object
    sp = {}
    start = time.time()

    @pydantic.validate_arguments
    def sparameter_calculation(
        n,
        component: Component,
        port_symmetries: Optional[PortSymmetries] = port_symmetries,
        monitor_indices: List[str] = monitor_indices,
        wavelength_start: float = wavelength_start,
        wavelength_stop: float = wavelength_stop,
        wavelength_points: int = wavelength_points,
        dirpath: Path = dirpath,
        animate: bool = animate,
        dispersive: bool = dispersive,
        **settings,
    ) -> Dict:
        """Return Sparameter dict."""

        sim_dict = get_simulation(
            component=component,
            port_source_name=f"o{monitor_indices[n]}",
            resolution=resolution,
            wavelength_start=wavelength_start,
            wavelength_stop=wavelength_stop,
            wavelength_points=wavelength_points,
            port_margin=port_margin,
            port_monitor_offset=port_monitor_offset,
            port_source_offset=port_source_offset,
            dispersive=dispersive,
            layer_stack=layer_stack,
            **settings,
        )

        sim = sim_dict["sim"]
        monitors = sim_dict["monitors"]
        # freqs = sim_dict["freqs"]
        # wavelengths = 1 / freqs
        # print(sim.resolution)

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
            animate.to_mp4(30, monitor_indices[n] + ".mp4")
        else:
            sim.run(until_after_sources=termination)
        # call this function every 50 time spes
        # look at simulation and measure Ez component
        # when field_monitor_point decays below a certain 1e-9 field threshold

        # Calculate mode overlaps
        # Get source monitor results
        component_ref = component.ref()
        source_entering, source_exiting = parse_port_eigenmode_coeff(
            monitor_indices[n], component_ref.ports, sim_dict
        )
        # Get coefficients
        for monitor_index in monitor_indices:
            j = monitor_indices[n]
            i = monitor_index
            if monitor_index == monitor_indices[n]:
                sii = source_exiting / source_entering
                siia = np.unwrap(np.angle(sii))
                siim = np.abs(sii)
                sp[f"s{i}{i}a"] = siia
                sp[f"s{i}{i}m"] = siim
            else:
                monitor_entering, monitor_exiting = parse_port_eigenmode_coeff(
                    monitor_index, component_ref.ports, sim_dict
                )
                sij = monitor_exiting / source_entering
                sija = np.unwrap(np.angle(sij))
                sijm = np.abs(sij)
                sp[f"s{i}{j}a"] = sija
                sp[f"s{i}{j}m"] = sijm
                sij = monitor_entering / source_entering
                sija = np.unwrap(np.angle(sij))
                sijm = np.abs(sij)

        if bool(port_symmetries) is True:
            for key in port_symmetries[f"o{monitor_indices[n]}"].keys():
                values = port_symmetries[f"o{monitor_indices[n]}"][key]
                for value in values:
                    sp[f"{value}m"] = sp[f"{key}m"]
                    sp[f"{value}a"] = sp[f"{key}a"]

        return sp

    # Since source is defined upon sim object instanciation, loop here
    # for port_index in monitor_indices:

    num_sims = len(port_symmetries.keys()) or len(source_indices)
    if lazy_parallelism:
        from mpi4py import MPI

        cores = min([num_sims, multiprocessing.cpu_count()])
        n = mp.divide_parallel_processes(cores)
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        sp = sparameter_calculation(
            n,
            component=component,
            port_symmetries=port_symmetries,
            wavelength_start=wavelength_start,
            wavelength_stop=wavelength_stop,
            wavelength_points=wavelength_points,
            animate=animate,
            monitor_indices=monitor_indices,
            **settings,
        )
        # Synchronize dicts
        if rank == 0:
            for i in range(1, size, 1):
                data = comm.recv(source=i, tag=11)
                sp.update(data)

            df = pd.DataFrame(sp)
            df["wavelengths"] = np.linspace(
                wavelength_start, wavelength_stop, wavelength_points
            )
            df["freqs"] = 1 / df["wavelengths"]
            df.to_csv(filepath, index=False)
            logger.info(f"Write simulation results to {filepath!r}")
            filepath_sim_settings.write_text(OmegaConf.to_yaml(sim_settings))
            logger.info(f"Write simulation settings to {filepath_sim_settings!r}")
            return df
        else:
            comm.send(sp, dest=0, tag=11)

    else:
        for n in tqdm(range(num_sims)):
            sp.update(
                sparameter_calculation(
                    n,
                    component=component,
                    port_symmetries=port_symmetries,
                    wavelength_start=wavelength_start,
                    wavelength_stop=wavelength_stop,
                    wavelength_points=wavelength_points,
                    animate=animate,
                    monitor_indices=monitor_indices,
                    **settings,
                )
            )
        df = pd.DataFrame(sp)
        df["wavelengths"] = np.linspace(
            wavelength_start, wavelength_stop, wavelength_points
        )
        df["freqs"] = 1 / df["wavelengths"]
        df.to_csv(filepath, index=False)

        end = time.time()
        sim_settings.update(compute_time_seconds=end - start)
        sim_settings.update(compute_time_minutes=(end - start) / 60)
        logger.info(f"Write simulation results to {filepath!r}")
        filepath_sim_settings.write_text(OmegaConf.to_yaml(sim_settings))
        logger.info(f"Write simulation settings to {filepath_sim_settings!r}")
        return df


write_sparameters_meep_1x1 = gf.partial(
    write_sparameters_meep, port_symmetries=port_symmetries.port_symmetries_1x1
)

write_sparameters_meep_1x1_bend90 = gf.partial(
    write_sparameters_meep,
    ymargin=0,
    ymargin_bot=3,
    xmargin_right=3,
    port_symmetries=port_symmetries.port_symmetries_1x1,
)

sig = inspect.signature(write_sparameters_meep)
settings_write_sparameters_meep = set(sig.parameters.keys()).union(
    settings_get_simulation
)

if __name__ == "__main__":
    c = gf.components.straight(length=2)
    write_sparameters_meep_1x1(c, run=False)

    # import matplotlib.pyplot as plt
    # plt.show()

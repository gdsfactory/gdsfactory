"""Compute and write Sparameters using Meep."""

from __future__ import annotations

import inspect
import multiprocessing
import pathlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import meep as mp
import numpy as np
import pydantic
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.config import logger
from gdsfactory.pdk import get_layer_stack
from gdsfactory.serialization import clean_value_json
from gdsfactory.simulation import port_symmetries
from gdsfactory.simulation.get_sparameters_path import (
    get_sparameters_path_meep as get_sparameters_path,
)
from gdsfactory.simulation.gmeep.get_simulation import (
    get_simulation,
    settings_get_simulation,
)
from gdsfactory.technology import LayerStack
from gdsfactory.typings import ComponentSpec, PathType, Port, PortSymmetries

ncores = multiprocessing.cpu_count()


def remove_simulation_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a copy of dict with only simulation settings.

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


def parse_port_eigenmode_coeff(port_name: str, ports: Dict[str, Port], sim_dict: Dict):
    """Returns the coefficients relative to whether the wavevector is entering or \
            exiting simulation.

    Args:
        port_index: index of port.
        ports: component_ref.ports.
        sim_dict: simulation dict.

    """
    if port_name not in ports:
        raise ValueError(f"port = {port_name!r} not in {list(ports.keys())}.")

    orientation = ports[port_name].orientation

    # Inputs
    sim = sim_dict["sim"]
    monitors = sim_dict["monitors"]

    # get_eigenmode_coeff.alpha[:,:,idx]
    # with ind being the forward or backward wave according to cell coordinates.
    # Figure out if that is exiting the simulation or not
    # depending on the port orientation (assuming it's near PMLs)
    if orientation == 0:  # east
        kpoint = mp.Vector3(x=1)
        idx_in = 1
        idx_out = 0
    elif orientation == 90:  # north
        kpoint = mp.Vector3(y=1)
        idx_in = 1
        idx_out = 0
    elif orientation == 180:  # west
        kpoint = mp.Vector3(x=1)
        idx_in = 0
        idx_out = 1
    elif orientation == 270:  # south
        kpoint = mp.Vector3(y=1)
        idx_in = 0
        idx_out = 1
    else:
        raise ValueError(
            f"Port orientation {orientation!r} not in 0, 90, 180, or 270 degrees!"
        )

    # Get port coeffs
    monitor_coeff = sim.get_eigenmode_coefficients(
        monitors[port_name], [1], kpoint_func=lambda f, n: kpoint
    )

    coeff_in = monitor_coeff.alpha[
        0, :, idx_in
    ]  # ingoing (w.r.t. simulation cell) wave
    coeff_out = monitor_coeff.alpha[
        0, :, idx_out
    ]  # outgoing (w.r.t. simulation cell) wave

    return coeff_in, coeff_out


@pydantic.validate_arguments
def write_sparameters_meep(
    component: ComponentSpec,
    port_source_names: Optional[List[str]] = None,
    port_symmetries: Optional[PortSymmetries] = None,
    resolution: int = 30,
    wavelength_start: float = 1.5,
    wavelength_stop: float = 1.6,
    wavelength_points: int = 50,
    dirpath: Optional[PathType] = None,
    layer_stack: Optional[LayerStack] = None,
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
    decay_by: float = 1e-3,
    is_3d: bool = False,
    z: float = 0,
    plot_args: Dict = None,
    **settings,
) -> Dict:
    r"""Returns Sparameters and writes them to npz filepath.

    Simulates each time using a different input port (by default, all of them)
    unless you specify port_symmetries:

    port_symmetries_crossing = {
        "o1@0,o1@0": ["o2@0,o2@0", "o3@0,o3@0", "o4@0,o4@0"],
        "o2@0,o1@0": ["o1@0,o2@0", "o3@0,o4@0", "o4@0,o3@0"],
        "o3@0,o1@0": ["o1@0,o3@0", "o2@0,o4@0", "o4@0,o2@0"],
        "o4@0,o1@0": ["o1@0,o4@0", "o2@0,o3@0", "o3@0,o2@0"],
    }

    - Only simulations using the outer key port names will be run
    - The associated value is another dict whose keys are the S-parameters computed
        when this source is active
    - The values of this inner Dict are lists of s-parameters whose values are copied


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
        resolution: in pixels/um (30: for coarse, 100: for fine).
        port_source_names: list of ports to excite. Defaults to all.
        port_symmetries: Dict to specify port symmetries, to save number of simulations.
        dirpath: directory to store Sparameters.
        layer_stack: contains layer to thickness, zmin and material.
            Defaults to active pdk.layer_stack.
        port_margin: margin on each side of the port.
        port_monitor_offset: offset between Component and monitor port in um.
        port_source_offset: offset between Component and source port in um.
        filepath: to store pandas Dataframe with Sparameters in npz format.
            Defaults to dirpath/component_.npz.
        overwrite: overwrites stored Sparameter npz results.
        animate: saves a MP4 images of the simulation for inspection, and also
            outputs during computation. The name of the file is the source index.
        lazy_parallelism: toggles the flag "meep.divide_parallel_processes" to
            perform the simulations with different sources in parallel.
            By default MPI just runs the same copy of the Python script everywhere,
            with the C++ under MEEP actually being parallelized.
            divide_parallel_processes allows us to logically split this one calculation
            into (in this case "cores") subdivisions.
            The only difference in the scripts is that a different integer n
            is returned depending on the subdivision it is running in.
            So we use that n to select different sources, and each subdivision calculates
            its own Sparams independently. Afterwards, we collect all
            results in one of the subdivisions (if rank == 0).
        run: runs simulation, if False, only plots simulation.
        dispersive: use dispersive models for materials (requires higher resolution).
        xmargin: left and right distance from component to PML.
        xmargin_left: west distance from component to PML.
        xmargin_right: east distance from component to PML.
        ymargin: top and bottom distance from component to PML.
        ymargin_top: north distance from component to PML.
        ymargin_bot: south distance from component to PML.
        is_3d: if True runs in 3D (much slower).
        z: for 2D plot.
        plot_args: if animate or not run, customization keyword arguments passed to
          `plot2D()` (i.e. `labels`, `eps_parameters`, `boundary_parameters`, `field_parameters`, etc.)

    keyword Args:
        extend_ports_length: to extend ports beyond the PML (um).
        zmargin_top: thickness for cladding above core (um).
        zmargin_bot: thickness for cladding below core (um).
        tpml: PML thickness (um).
        clad_material: material for cladding.
        wavelength_start: wavelength min (um).
        wavelength_stop: wavelength max (um).
        wavelength_points: wavelength steps.
        dfcen: delta frequency.
        port_source_name: input port name.
        port_margin: margin on each side of the port (um).
        distance_source_to_monitors: in (um).
        port_source_offset: offset between source Component port and source MEEP port.
        port_monitor_offset: offset between Component and MEEP port monitor.
        material_name_to_meep: map layer_stack names with meep material database name
            or refractive index. dispersive materials have a wavelength dependent index.

    Returns:
        sparameters in a Dict (wavelengths, s11a, o1@0,o2@0, ...)
            where `a` is the angle in radians and `m` the module.

    """
    component = (
        component if isinstance(component, Component) else gf.get_component(component)
    )
    layer_stack = layer_stack or get_layer_stack()

    plot_args = plot_args or {}

    for setting in settings:
        if setting not in settings_get_simulation:
            raise ValueError(f"{setting!r} not in {settings_get_simulation}")

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
        is_3d=is_3d,
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
            dispersive=dispersive,
            is_3d=is_3d,
            **settings,
        )
        sim = sim_dict["sim"]
        if is_3d:
            sim.plot2D(
                output_plane=mp.Volume(
                    size=mp.Vector3(sim.cell_size.x, sim.cell_size.y, 0),
                    center=mp.Vector3(0, 0, z),
                ),
                **plot_args,
            )
        else:
            sim.plot2D(plot_eps_flag=True, **plot_args)
        return sim

    if filepath.exists():
        if not overwrite:
            logger.info(f"Simulation loaded from {filepath!r}")
            return dict(np.load(filepath))
        elif overwrite:
            filepath.unlink()

    component_ref = component.ref()
    ports = component_ref.ports
    port_names = [port.name for port in list(ports.values())]
    port_source_names = port_source_names or port_names
    num_sims = len(port_source_names) - len(port_symmetries)

    sp = {}  # Sparameters dict
    start = time.time()

    @pydantic.validate_arguments
    def sparameter_calculation(
        port_source_name: str,
        component: Component,
        port_symmetries: Optional[PortSymmetries] = port_symmetries,
        port_names: List[str] = port_names,
        wavelength_start: float = wavelength_start,
        wavelength_stop: float = wavelength_stop,
        wavelength_points: int = wavelength_points,
        animate: bool = animate,
        plot_args: Dict = plot_args,
        dispersive: bool = dispersive,
        decay_by: float = decay_by,
        **settings,
    ) -> Dict:
        """Return Sparameter dict."""
        sim_dict = get_simulation(
            component=component,
            port_source_name=port_source_name,
            resolution=resolution,
            wavelength_start=wavelength_start,
            wavelength_stop=wavelength_stop,
            wavelength_points=wavelength_points,
            port_margin=port_margin,
            port_monitor_offset=port_monitor_offset,
            port_source_offset=port_source_offset,
            dispersive=dispersive,
            layer_stack=layer_stack,
            is_3d=is_3d,
            **settings,
        )

        sim = sim_dict["sim"]
        # freqs = sim_dict["freqs"]
        # wavelengths = 1 / freqs
        # print(sim.resolution)

        # Terminate when the area in the whole area decayed
        termination = [mp.stop_when_energy_decayed(dt=50, decay_by=decay_by)]

        if animate:
            # Defaults for animation
            if "field_parameters" not in plot_args:
                plot_args["field_parameters"] = {
                    "alpha": 0.8,
                    "cmap": "RdBu",
                    "interpolation": "none",
                }
            if "eps_parameters" not in plot_args:
                plot_args["eps_parameters"] = {"contour": True}
            if "fields" not in plot_args:
                plot_args["fields"] = mp.Ez
            if "realtime" not in plot_args:
                plot_args["realtime"] = True
            if "normalize" not in plot_args:
                plot_args["normalize"] = True

            sim.use_output_directory()
            animate = mp.Animate2D(
                sim,
                **plot_args,
            )
            sim.run(mp.at_every(1, animate), until_after_sources=termination)
            animate.to_mp4(30, f"{component.name}_{port_source_name}.mp4")
        else:
            sim.run(until_after_sources=termination)

        # Calculate mode overlaps
        # Get source monitor results
        source_entering, _ = parse_port_eigenmode_coeff(
            port_source_name, component.ports, sim_dict
        )
        # Get coefficients
        for port_name in port_names:
            _, monitor_exiting = parse_port_eigenmode_coeff(
                port_name, component.ports, sim_dict
            )
            key = f"{port_name}@0,{port_source_name}@0"
            sp[key] = monitor_exiting / source_entering

        if bool(port_symmetries):
            for key, symmetries in port_symmetries.items():
                for sym in symmetries:
                    if key in sp:
                        sp[sym] = sp[key]

        return sp

    if lazy_parallelism:
        from mpi4py import MPI

        cores = min([num_sims, multiprocessing.cpu_count()])

        n = mp.divide_parallel_processes(cores)
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # Map port names to integers
        port_source_dict = {}
        for number, name in enumerate(port_source_names):
            port_source_dict[number] = name

        sp = sparameter_calculation(
            port_source_name=port_source_dict[n],
            component=component,
            port_symmetries=port_symmetries,
            wavelength_start=wavelength_start,
            wavelength_stop=wavelength_stop,
            wavelength_points=wavelength_points,
            animate=animate,
            port_names=port_names,
            **settings,
        )
        # Synchronize dicts
        if rank == 0:
            for i in range(1, size):
                data = comm.recv(source=i, tag=11)
                sp.update(data)

            sp["wavelengths"] = np.linspace(
                wavelength_start, wavelength_stop, wavelength_points
            )
            np.savez_compressed(filepath, **sp)
            logger.info(f"Write simulation results to {filepath!r}")
            filepath_sim_settings.write_text(
                OmegaConf.to_yaml(clean_value_json(sim_settings))
            )
            logger.info(f"Write simulation settings to {filepath_sim_settings!r}")
            return sp
        else:
            comm.send(sp, dest=0, tag=11)

    else:
        for port_source_name in tqdm(port_source_names):
            sp.update(
                sparameter_calculation(
                    port_source_name,
                    component=component,
                    port_symmetries=port_symmetries,
                    wavelength_start=wavelength_start,
                    wavelength_stop=wavelength_stop,
                    wavelength_points=wavelength_points,
                    animate=animate,
                    port_names=port_names,
                    **settings,
                )
            )
        sp["wavelengths"] = np.linspace(
            wavelength_start, wavelength_stop, wavelength_points
        )
        np.savez_compressed(filepath, **sp)

        end = time.time()
        sim_settings.update(compute_time_seconds=end - start)
        sim_settings.update(compute_time_minutes=(end - start) / 60)
        logger.info(f"Write simulation results to {filepath!r}")
        filepath_sim_settings.write_text(OmegaConf.to_yaml(sim_settings))
        logger.info(f"Write simulation settings to {filepath_sim_settings!r}")
        return sp


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
    wavelength_start = 1.26
    wavelength_stop = 1.36
    sim_settings = dict(
        wavelength_start=wavelength_start, wavelength_stop=wavelength_stop
    )
    # c = gf.components.mmi1x2(cross_section=gf.cross_section.strip)
    c = gf.components.straight(length=2)
    import matplotlib.pyplot as plt

    def func(x):
        result = np.where(np.abs(x) > 1e-10, np.abs(x) ** 2, -10)
        return np.log10(result, out=result, where=result > 0)

    sp = write_sparameters_meep(
        c,
        run=True,
        animate=True,
        is_3d=False,
        plot_args={
            "eps_parameters": {"contour": True},
            "field_parameters": {
                "alpha": 0.8,
                "cmap": "RdBu",
                "interpolation": "none",
                "post_process": func,
            },
            "realtime": False,
        },
        overwrite=True,
        **sim_settings,
    )
    plt.show()

    # from gdsfactory.simulation.add_simulation_markers import add_simulation_markers
    # import gdsfactory.simulation as sim

    # c = gf.components.straight(length=2)
    # c = gf.components.bend_euler(radius=3)
    # c = add_simulation_markers(c)

    # sp = write_sparameters_meep_1x1(c, run=True, is_3d=False)
    # sim.plot.plot_sparameters(sp)

    # import matplotlib.pyplot as plt
    # plt.show()

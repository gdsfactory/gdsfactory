"""Compute and write Sparameters using Meep.

Synchronize dicts
from https://stackoverflow.com/questions/66703153/updating-dictionary-values-in-mpi4py
"""

import multiprocessing
import pathlib
import pickle
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd
import pydantic
from omegaconf import OmegaConf

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.config import CONFIG, logger
from gdsfactory.simulation.get_sparameters_path import get_sparameters_path
from gdsfactory.simulation.gmeep.get_simulation import get_simulation
from gdsfactory.tech import LAYER_STACK, LayerStack

ncores = multiprocessing.cpu_count()


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
    component: Component,
    resolution: int = 20,
    wl_min: float = 1.5,
    wl_max: float = 1.6,
    wl_steps: int = 50,
    dirpath: Path = CONFIG["sparameters"],
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
    **settings,
) -> pd.DataFrame:
    """Compute Sparameters and writes them in CSV filepath.
    Repeats the simulation, each time using a different input port sequentially
    (by default, all of them)

    TODO: provide list of port name tuples whose results to merge (e.g. symmetric ports)

    Args:
        component: to simulate.
        resolution: in pixels/um (20: for coarse, 120: for fine)
        source_ports: list of port string names to use as sources
        dirpath: directory to store Sparameters
        layer_stack:
        port_margin: margin on each side of the port
        port_monitor_offset: offset between monitor GDS port and monitor MEEP port
        port_source_offset: offset between source GDS port and source MEEP port
        filepath: to store pandas Dataframe with Sparameters in CSV format.
            Defaults to dirpath/component_.csv
        overwrite: overwrites
        animate: saves a MP4 images of the simulation for inspection, and also
            outputs during computation. The name of the file is the source index
        lazy_parallelism: toggles the flag "meep.divide_parallel_processes" to
            perform the simulations with different sources in parallel
        dispersive: use dispersive models for materials (requires higher resolution)

    keyword Args:
        layer_to_thickness: GDS layer (int, int) to thickness
        layer_to_material: GDS layer (int, int) to material string ('Si', 'SiO2', ...)
        extend_ports_length: to extend ports beyond the PML
        layer_stack: Dict of layer number (int, int) to thickness (um)
        t_clad_top: thickness for cladding above core
        t_clad_bot: thickness for cladding below core
        tpml: PML thickness (um)
        clad_material: material for cladding
        is_3d: if True runs in 3D
        wl_min: wavelength min (um)
        wl_max: wavelength max (um)
        wl_steps: wavelength steps
        dfcen: delta frequency
        port_source_name: input port name
        port_field_monitor_name:
        port_margin: margin on each side of the port
        distance_source_to_monitors: in (um) source goes before
        port_source_offset: offset between source GDS port and source MEEP port
        port_monitor_offset: offset between monitor GDS port and monitor MEEP port

    Returns:
        sparameters in a pandas Dataframe

    """
    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_material = layer_stack.get_layer_to_material()
    # layer_to_zmin = layer_stack.get_layer_to_zmin()

    sim_settings = dict(
        resolution=resolution,
        wl_min=wl_min,
        wl_max=wl_max,
        wl_steps=wl_steps,
        port_margin=port_margin,
        port_monitor_offset=port_monitor_offset,
        port_source_offset=port_source_offset,
        dispersive=dispersive,
        **settings,
    )

    filepath = filepath or get_sparameters_path(
        component=component,
        dirpath=dirpath,
        suffix=".csv",
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

    if not run:
        sim_dict = get_simulation(
            component=component,
            wl_min=wl_min,
            wl_max=wl_max,
            wl_steps=wl_steps,
            layer_stack=layer_stack,
            port_margin=port_margin,
            port_monitor_offset=port_monitor_offset,
            port_source_offset=port_source_offset,
            resolution=20,
            **settings,
        )
        sim_dict["sim"].plot2D()
        plt.show()
        return

    if filepath.exists() and not overwrite:
        logger.info(f"Simulation loaded from {filepath!r}")
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
        wl_min: float = wl_min,
        wl_max: float = wl_max,
        wl_steps: int = wl_steps,
        dirpath: Path = dirpath,
        layer_to_thickness: Dict[Tuple[int, int], float] = layer_to_thickness,
        layer_to_material: Dict[Tuple[int, int], str] = layer_to_material,
        animate: bool = animate,
        dispersive: bool = dispersive,
        **settings,
    ) -> Dict:

        sim_dict = get_simulation(
            component=component,
            port_source_name=f"o{Sparams_indices[n]}",
            resolution=resolution,
            wl_min=wl_min,
            wl_max=wl_max,
            wl_steps=wl_steps,
            port_margin=port_margin,
            port_monitor_offset=port_monitor_offset,
            port_source_offset=port_source_offset,
            dispersive=dispersive,
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
            animate.to_mp4(30, Sparams_indices[n] + ".mp4")
        else:
            sim.run(until_after_sources=termination)
        # call this function every 50 time spes
        # look at simulation and measure Ez component
        # when field_monitor_point decays below a certain 1e-9 field threshold

        # Calculate mode overlaps
        # Get source monitor results
        source_entering, source_exiting = parse_port_eigenmode_coeff(
            Sparams_indices[n], component_ref.ports, sim_dict
        )
        # Get coefficients
        for monitor_index in Sparams_indices:
            i = Sparams_indices[n]
            j = monitor_index
            if monitor_index == Sparams_indices[n]:
                sii = source_exiting / source_entering
                siia = np.unwrap(np.angle(sii))
                siim = np.abs(sii)
                # Sparams_dict[f"sourceExiting{i}{j}"] = source_exiting
                # Sparams_dict[f"sourceEntering{i}{j}"] = source_entering
                Sparams_dict[f"s{i}{j}a"] = siia
                Sparams_dict[f"s{i}{j}m"] = siim
            else:
                monitor_entering, monitor_exiting = parse_port_eigenmode_coeff(
                    monitor_index, component_ref.ports, sim_dict
                )
                sij = monitor_exiting / source_entering
                sija = np.unwrap(np.angle(sij))
                sijm = np.abs(sij)
                # Sparams_dict[f"monitorExiting{i}{j}"] = monitor_exiting
                # Sparams_dict[f"monitorEntering{i}{j}"] = monitor_entering
                Sparams_dict[f"s{i}{j}a"] = sija
                Sparams_dict[f"s{i}{j}m"] = sijm
                sij = monitor_entering / source_entering
                sija = np.unwrap(np.angle(sij))
                sijm = np.abs(sij)
                # Sparams_dict[f"s{i}{j}a_enter"] = sija
                # Sparams_dict[f"s{i}{j}m_enter"] = sijm

        return Sparams_dict

    # Since source is defined upon sim object loop here
    # for port_index in Sparams_indices:

    if lazy_parallelism:
        import multiprocessing

        from mpi4py import MPI

        cores = min([len(Sparams_indices), multiprocessing.cpu_count()])
        # mp.count_processors = lambda x: cores
        # FIXME RuntimeError: meep: numgroups > count_processors
        # Using MPI version 3.1, 1 processes

        n = mp.divide_parallel_processes(cores)
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
        # Synchronize dicts
        if rank == 0:
            for i in range(1, size, 1):
                data = comm.recv(source=i, tag=11)
                Sparams_dict.update(data)

            df = pd.DataFrame(Sparams_dict)
            df["wavelengths"] = np.linspace(wl_min, wl_max, wl_steps)
            df["freqs"] = 1 / df["wavelengths"]
            df.to_csv(filepath, index=False)
            logger.info(f"Write simulation results to {filepath!r}")
            filepath_sim_settings.write_text(OmegaConf.to_yaml(sim_settings))
            logger.info(f"Write simulation settings to {filepath_sim_settings!r}")
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

        logger.info(f"Write simulation results to {filepath!r}")
        filepath_sim_settings.write_text(OmegaConf.to_yaml(sim_settings))
        logger.info(f"Write simulation settings to {filepath_sim_settings!r}")
        return df


@pydantic.validate_arguments
def write_sparameters_meep_mpi(
    component: Component,
    cores: int = ncores,
    filepath: Optional[Path] = None,
    dirpath: Path = CONFIG["sparameters"],
    layer_stack: LayerStack = LAYER_STACK,
    temp_dir: Path = CONFIG["sparameters"] / "temp",
    temp_file_str: str = "write_sparameters_meep_mpi",
    overwrite: bool = False,
    **kwargs,
) -> Path:
    """Write Sparameters using multiple cores and MPI
    and returns Sparameters filepath.

    Args:
        component: gdsfactory Component.
        cores: number of processors.
        temp_dir: temporary directory to hold simulation files.
        filepath: to store pandas Dataframe with Sparameters in CSV format.
            Defaults to dirpath/component_.csv
        dirpath: directory to store Sparameters
        layer_stack:
        temp_file_str: names of temporary files in temp_dir.

    Keyword Args:
        resolution: in pixels/um (20: for coarse, 120: for fine)
        source_ports: list of port string names to use as sources
        dirpath: directory to store Sparameters
        layer_to_thickness: GDS layer (int, int) to thickness
        layer_to_material: GDS layer (int, int) to material string ('Si', 'SiO2', ...)
        port_margin: margin on each side of the port
        port_monitor_offset: offset between monitor GDS port and monitor MEEP port
        port_source_offset: offset between source GDS port and source MEEP port
        filepath: to store pandas Dataframe with Sparameters in CSV format.
            Defaults to dirpath/component_.csv
        overwrite: overwrites
        animate: saves a MP4 images of the simulation for inspection, and also
            outputs during computation. The name of the file is the source index
        lazy_parallelism: toggles the flag "meep.divide_parallel_processes" to
            perform the simulations with different sources in parallel
        dispersive: use dispersive models for materials (requires higher resolution)
        extend_ports_length: to extend ports beyond the PML
        layer_stack: Dict of layer number (int, int) to thickness (um)
        t_clad_top: thickness for cladding above core
        t_clad_bot: thickness for cladding below core
        tpml: PML thickness (um)
        clad_material: material for cladding
        is_3d: if True runs in 3D
        wl_min: wavelength min (um)
        wl_max: wavelength max (um)
        wl_steps: wavelength steps
        dfcen: delta frequency
        port_source_name: input port name
        port_field_monitor_name:
        port_margin: margin on each side of the port
        distance_source_to_monitors: in (um) source goes before
        port_source_offset: offset between source GDS port and source MEEP port
        port_monitor_offset: offset between monitor GDS port and monitor MEEP port
    """
    filepath_df = filepath or get_sparameters_path(
        component=component,
        dirpath=dirpath,
        suffix=".csv",
        layer_stack=layer_stack,
        **kwargs,
    )
    if filepath_df.exists() and not overwrite:
        logger.info(f"Simulation {filepath_df!r} already exists")
        return filepath_df

    # Save the component object to simulation for later retrieval
    temp_dir.mkdir(exist_ok=True, parents=True)
    filepath = temp_dir / temp_file_str
    component_file = filepath.with_suffix(".pkl")
    kwargs.update(filepath=str(filepath_df))

    with open(component_file, "wb") as outp:
        pickle.dump(component, outp, pickle.HIGHEST_PROTOCOL)

    # Write execution file
    script_lines = [
        "import pickle\n",
        "from gdsfactory.simulation.gmeep import write_sparameters_meep\n\n",
        'if __name__ == "__main__":\n\n',
        f"\twith open(\"{component_file}\", 'rb') as inp:\n",
        "\t\tcomponent = pickle.load(inp)\n\n"
        "\twrite_sparameters_meep(component = component,\n",
    ]
    for key in kwargs.keys():
        script_lines.append(f"\t\t{key} = {kwargs[key]!r},\n")

    script_lines.append("\t)")
    script_file = filepath.with_suffix(".py")
    script_file_obj = open(script_file, "w")
    script_file_obj.writelines(script_lines)
    script_file_obj.close()

    command = f"mpirun -np {cores} python {script_file}"
    logger.info(command)
    logger.info(str(filepath_df))

    subprocess.Popen(
        shlex.split(command),
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return filepath_df


@pydantic.validate_arguments
def write_sparameters_meep_mpi_pool(
    jobs: List[Dict],
    cores_per_run: int = 2,
    total_cores: int = 4,
    temp_dir: Path = CONFIG["sparameters"] / "temp",
    delete_temp_files: bool = True,
):
    """Write Sparameters
    Given a tuple of write_sparameters_meep keyword arguments (the "jobs"),
        launches parallel simulations
    Each simulation is assigned "cores_per_run" cores
    A total of "total_cores" is assumed, if cores_per_run * len(jobs) > total_cores
    then the overflow will be performed serially

    Args
        jobs: list of Dicts containing the simulation settings for each job.
            for write_sparameters_meep
        cores_per_run: number of processors to assign to each component simulation
        total_cores: total number of cores to use
        temp_dir: temporary directory to hold simulation files
        delete_temp_file (Boolean): whether to delete temp_dir when done
        verbose: log progress messages
    """

    # Setup pools
    num_pools = int(np.ceil(cores_per_run * len(jobs) / total_cores))
    jobs_per_pool = int(np.floor(total_cores / cores_per_run))
    njobs = len(jobs)

    logger.info(f"Running parallel simulations over {njobs} jobs")
    logger.info(
        f"Using a total of {total_cores} cores with {cores_per_run} cores per job"
    )
    logger.info(
        f"Tasks split amongst {num_pools} pools with up to {jobs_per_pool} jobs each."
    )

    i = 0
    # For each pool
    for j in range(num_pools):
        processes = []

        # For each job in the pool
        for k in range(jobs_per_pool):
            # Flag to catch non full pools
            if i >= njobs:
                continue
            logger.info(f"Task {k} of pool {j} is job {i}")

            # Obtain current job
            simulations_settings = jobs[i]

            process = write_sparameters_meep_mpi(
                cores=cores_per_run,
                temp_dir=temp_dir,
                temp_file_str=f"write_sparameters_meep_mpi_{i}",
                **simulations_settings,
            )
            processes.append(process)

            # Increment task number
            i += 1

        # Wait for pool to end
        for process in processes:
            process.wait()

    if delete_temp_files:
        shutil.rmtree(temp_dir)


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
    # c = gf.components.crossing()

    # Multicore example

    c1 = gf.c.straight(length=5)
    p = 3
    c1 = gf.add_padding_container(c1, default=0, top=p, bottom=p)
    proc = write_sparameters_meep_mpi(
        component=c1,
        cores=3,
    )

    # Multicore pools example
    c1 = gf.c.straight(length=5)
    p = 3
    c1 = gf.add_padding_container(c1, default=0, top=p, bottom=p)

    c2 = gf.c.straight(length=4)
    p = 3
    c2 = gf.add_padding_container(c2, default=0, top=p, bottom=p)

    c1_dict = {
        "component": c1,
        "run": True,
        "overwrite": True,
        "lazy_parallelism": True,
        "filepath": "c1_dict.csv",
    }
    c2_dict = {
        "component": c2,
        "run": True,
        "overwrite": True,
        "lazy_parallelism": True,
        "filepath": "c2_dict.csv",
    }
    c3_dict = {
        "component": c2,
        "run": True,
        "overwrite": True,
        "lazy_parallelism": True,
        "resolution": 40,
        "port_source_offset": 0.3,
        "filepath": "c3_dict.csv",
    }

    # jobs
    jobs = [
        c1_dict,
        c2_dict,
        c3_dict,
    ]

    write_sparameters_meep_mpi_pool(
        jobs=jobs,
        cores_per_run=4,
        total_cores=10,
        delete_temp_files=False,
    )

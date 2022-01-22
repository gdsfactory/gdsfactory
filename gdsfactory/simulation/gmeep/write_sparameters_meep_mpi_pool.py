"""Compute and write Sparameters using Meep in an MPI pool."""

import multiprocessing
import pathlib
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pydantic

import gdsfactory as gf
from gdsfactory.config import CONFIG, logger
from gdsfactory.simulation.get_sparameters_path import (
    get_sparameters_path_meep as get_sparameters_path,
)
from gdsfactory.simulation.gmeep.write_sparameters_meep import remove_simulation_kwargs
from gdsfactory.simulation.gmeep.write_sparameters_meep_mpi import (
    write_sparameters_meep_mpi,
)
from gdsfactory.tech import LAYER_STACK, LayerStack

ncores = multiprocessing.cpu_count()


@pydantic.validate_arguments
def write_sparameters_meep_mpi_pool(
    jobs: List[Dict],
    cores_per_run: int = 2,
    total_cores: int = 4,
    temp_dir: Path = CONFIG["sparameters"] / "temp",
    delete_temp_files: bool = True,
    dirpath: Path = CONFIG["sparameters"],
    layer_stack: LayerStack = LAYER_STACK,
    **kwargs,
) -> List[Path]:
    """Write Sparameters and returns the filepaths
    Given a list of write_sparameters_meep keyword arguments (the "jobs"),
        launches them in different cores
    Each simulation is assigned "cores_per_run" cores
    A total of "total_cores" is assumed, if cores_per_run * len(jobs) > total_cores
    then the overflow will run sequentially (not in parallel)

    Args
        jobs: list of Dicts containing the simulation settings for each job.
            for write_sparameters_meep
        cores_per_run: number of processors to assign to each component simulation
        total_cores: total number of cores to use
        temp_dir: temporary directory to hold simulation files
        delete_temp_files: deletes temp_dir when done
        dirpath: directory to store Sparameters
        layer_stack:

    keyword Args:
        overwrite: overwrites stored simulation results.
        dispersive: use dispersive models for materials (requires higher resolution)
        extend_ports_length: to extend ports beyond the PML
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
        filepath list for sparameters CSV (wavelengths, s11a, s12m, ...)
            where `a` is the angle in radians and `m` the module

    """
    # Parse jobs
    jobs_to_run = []
    for job in jobs:
        settings = remove_simulation_kwargs(kwargs)
        filepath = job.get(
            "filepath",
            get_sparameters_path(
                component=job["component"],
                dirpath=dirpath,
                layer_stack=layer_stack,
                **settings,
            ),
        )
        if filepath.exists():
            job.update(**kwargs)
            if job.get("overwrite", False):
                pathlib.Path.unlink(filepath)
                logger.info(
                    f"Simulation {filepath!r} found and overwrite is True. "
                    "Deleting file and adding it to the queue."
                )
                jobs_to_run.append(job)
            else:
                logger.info(
                    f"Simulation {filepath!r} found exists and "
                    "overwrite is False. Removing it from the queue."
                )
        else:
            logger.info(f"Simulation {filepath!r} not found. Adding it to the queue")
            jobs_to_run.append(job)

    # Update jobs
    jobs = jobs_to_run

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
        filepaths = []

        # For each job in the pool
        for k in range(jobs_per_pool):
            # Flag to catch non full pools
            if i >= njobs:
                continue
            logger.info(f"Task {k} of pool {j} is job {i}")

            # Obtain current job
            simulations_settings = jobs[i]

            filepath = write_sparameters_meep_mpi(
                cores=cores_per_run,
                temp_dir=temp_dir,
                temp_file_str=f"write_sparameters_meep_mpi_{i}",
                wait_to_finish=False,
                **simulations_settings,
            )
            filepaths.append(filepath)

            # Increment task number
            i += 1

        # Wait for pool to end
        done = False
        num_pool_jobs = len(filepaths)
        while not done:
            # Check if all jobs finished
            jobs_done = 0
            for filepath in filepaths:
                if filepath.exists():
                    jobs_done += 1
            if jobs_done == num_pool_jobs:
                done = True
            else:
                time.sleep(1)

    if delete_temp_files:
        shutil.rmtree(temp_dir)
    return filepaths


if __name__ == "__main__":

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
        "filepath": Path("c1_dict.csv"),
    }
    c2_dict = {
        "component": c2,
        "run": True,
        "overwrite": False,
        "lazy_parallelism": True,
        "filepath": Path("c2_dict.csv"),
    }
    c3_dict = {
        "component": c2,
        "run": True,
        "overwrite": True,
        "lazy_parallelism": True,
        "resolution": 40,
        "port_source_offset": 0.3,
        "filepath": Path("c3_dict.csv"),
    }

    # jobs
    jobs = [
        c1_dict,
        c2_dict,
        c3_dict,
    ]

    filepaths = write_sparameters_meep_mpi_pool(
        jobs=jobs,
        cores_per_run=4,
        total_cores=10,
        delete_temp_files=False,
    )

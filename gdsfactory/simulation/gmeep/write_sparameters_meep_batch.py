"""Compute and write Sparameters using Meep in an MPI pool."""

from __future__ import annotations

import multiprocessing
import pathlib
import shutil
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional

import numpy as np
import pydantic
from tqdm.auto import tqdm

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.config import logger, sparameters_path
from gdsfactory.pdk import get_layer_stack
from gdsfactory.simulation import port_symmetries
from gdsfactory.simulation.get_sparameters_path import (
    get_sparameters_path_meep as get_sparameters_path,
)
from gdsfactory.simulation.gmeep.write_sparameters_meep import remove_simulation_kwargs
from gdsfactory.simulation.gmeep.write_sparameters_meep_mpi import (
    write_sparameters_meep_mpi,
)
from gdsfactory.technology import LayerStack

ncores = multiprocessing.cpu_count()

temp_dir_default = Path(sparameters_path) / "temp"


@pydantic.validate_arguments
def write_sparameters_meep_batch(
    jobs: List[Dict],
    cores_per_run: int = 2,
    total_cores: int = 4,
    temp_dir: Path = temp_dir_default,
    delete_temp_files: bool = True,
    dirpath: Optional[Path] = None,
    layer_stack: Optional[LayerStack] = None,
    **kwargs,
) -> List[Path]:
    """Write Sparameters for a batch of jobs using MPI and returns results filepaths.

    Given a list of write_sparameters_meep keyword arguments `jobs` launches them in
    different cores using MPI where each simulation runs with `cores_per_run` cores.
    If there are more simulations than cores each batch runs sequentially.


    Args
        jobs: list of Dicts containing the simulation settings for each job.
            for write_sparameters_meep.
        cores_per_run: number of processors to assign to each component simulation.
        total_cores: total number of cores to use.
        temp_dir: temporary directory to hold simulation files.
        delete_temp_files: deletes temp_dir when done.
        dirpath: directory to store Sparameters.
        layer_stack: contains layer to thickness, zmin and material.
            Defaults to active pdk.layer_stack.

    keyword Args:
        resolution: in pixels/um (30: for coarse, 100: for fine).
        port_symmetries: Dict to specify port symmetries, to save number of simulations.
        dirpath: directory to store Sparameters.
        port_margin: margin on each side of the port.
        port_monitor_offset: offset between monitor GDS port and monitor MEEP port.
        port_source_offset: offset between source GDS port and source MEEP port.
        filepath: to store pandas Dataframe with Sparameters in CSV format..
        animate: saves a MP4 images of the simulation for inspection, and also
            outputs during computation. The name of the file is the source index.
        lazy_parallelism: toggles the flag "meep.divide_parallel_processes" to
            perform the simulations with different sources in parallel.
        dispersive: use dispersive models for materials (requires higher resolution).
        xmargin: left and right distance from component to PML.
        xmargin_left: west distance from component to PML.
        xmargin_right: east distance from component to PML.
        ymargin: top and bottom distance from component to PML.
        ymargin_top: north distance from component to PML.
        ymargin_bot: south distance from component to PML.
        extend_ports_length: to extend ports beyond the PML
        layer_stack: Dict of layer number (int, int) to thickness (um).
        zmargin_top: thickness for cladding above core.
        zmargin_bot: thickness for cladding below core.
        tpml: PML thickness (um).
        clad_material: material for cladding.
        is_3d: if True runs in 3D.
        wavelength_start: wavelength min (um).
        wavelength_stop: wavelength max (um).
        wavelength_points: wavelength steps.
        dfcen: delta frequency.
        port_source_name: input port name.
        port_margin: margin on each side of the port.
        distance_source_to_monitors: in (um) source goes before.
        port_source_offset: offset between source GDS port and source MEEP port.
        port_monitor_offset: offset between monitor GDS port and monitor MEEP port.

    Returns:
        filepath list for sparameters numpy saved files (wavelengths, o1@0,o2@0, ...).

    """
    layer_stack = layer_stack or get_layer_stack()

    # Parse jobs
    jobs_to_run = []
    for job in jobs:
        component = job["component"]
        component = gf.get_component(component)
        assert isinstance(component, Component)
        settings = remove_simulation_kwargs(kwargs)
        filepath = job.get(
            "filepath",
            get_sparameters_path(
                component=component,
                dirpath=dirpath,
                layer_stack=layer_stack,
                **settings,
            ),
        )
        if filepath.exists():
            job.update(**kwargs)
            if job.get("overwrite", kwargs.get("overwrite", False)):
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

    jobs = jobs_to_run

    batches = int(np.ceil(cores_per_run * len(jobs) / total_cores))
    jobs_per_batch = int(np.floor(total_cores / cores_per_run))
    njobs = len(jobs)
    logger.info(f"Running {njobs} simulations")
    logger.info(f"total_cores = {total_cores} with cores_per_run = {cores_per_run}")
    logger.info(f"Running {batches} batches with up to {jobs_per_batch} jobs each.")

    i = 0
    # For each batch in the pool
    for j in tqdm(range(batches)):
        filepaths = []

        # For each job in the batch
        for k in range(jobs_per_batch):
            if i >= njobs:
                continue
            logger.info(f"Task {k} of batch {j} is job {i}")

            # Obtain current job
            simulations_settings = jobs[i]

            pprint(simulations_settings)

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

        # Wait for batch to end
        done = False
        num_pool_jobs = len(filepaths)
        while not done:
            # Check if all jobs finished
            jobs_done = sum(1 for filepath in filepaths if filepath.exists())
            if jobs_done == num_pool_jobs:
                done = True
            else:
                time.sleep(1)

    temp_dir = pathlib.Path(temp_dir)
    if temp_dir.exists() and delete_temp_files:
        shutil.rmtree(temp_dir)
    return filepaths


write_sparameters_meep_batch_1x1 = gf.partial(
    write_sparameters_meep_batch, port_symmetries=port_symmetries.port_symmetries_1x1
)

write_sparameters_meep_batch_1x1_bend90 = gf.partial(
    write_sparameters_meep_batch,
    port_symmetries=port_symmetries.port_symmetries_1x1,
    ymargin=0,
    ymargin_bot=3,
    xmargin_right=3,
)


if __name__ == "__main__":
    jobs = [
        {
            "component": gf.components.straight(length=i),
            "run": True,
            "overwrite": True,
            "lazy_parallelism": False,
            "ymargin": 3,
        }
        for i in range(1, 4)
    ]

    filepaths = write_sparameters_meep_batch(
        jobs=jobs,
        cores_per_run=4,
        total_cores=8,
    )

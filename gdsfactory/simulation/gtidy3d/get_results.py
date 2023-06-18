"""Tidy3D."""

from __future__ import annotations

import pathlib
import concurrent.futures
import hashlib
from typing import Awaitable

import tidy3d as td
from tidy3d import web
from tidy3d.exceptions import WebError

import gdsfactory as gf
from gdsfactory.config import PATH, logger
from gdsfactory.typings import PathType

_executor = concurrent.futures.ThreadPoolExecutor()


def get_sim_hash(sim: td.Simulation) -> str:
    """Returns simulation hash as the unique ID."""
    return hashlib.md5(str(sim).encode()).hexdigest()


def _get_results(
    sim: td.Simulation,
    dirpath: PathType = PATH.results_tidy3d,
    overwrite: bool = False,
    verbose: bool = False,
) -> td.SimulationData:
    """Return SimulationData results from simulation.

    Only submits simulation if results not found locally or remotely.
    First tries to load simulation results from disk.
    Then it tries to load them from the server storage.
    Finally, submits simulation to run remotely

    Args:
        sim: tidy3d Simulation.
        dirpath: to store results locally.
        overwrite: overwrites the data even when path exists.
        verbose: prints info messages and progressbars.
    """
    sim_hash = get_sim_hash(sim)
    dirpath = pathlib.Path(dirpath)
    filename = f"{sim_hash}.hdf5"
    filepath = dirpath / filename

    # Look for results in local storage
    if filepath.exists():
        logger.info(f"Simulation results for {sim_hash!r} found in {filepath}")
        return td.SimulationData.from_file(str(filepath))

    # Look for results in tidy3d server storage
    hash_to_id = {d["taskName"]: d["task_id"] for d in web.get_tasks()}

    if sim_hash in hash_to_id:
        task_id = hash_to_id[sim_hash]
        web.monitor(task_id)

        try:
            return web.load(task_id=task_id, path=filename, replace_existing=overwrite)
        except WebError:
            print(f"task_id {task_id!r} exists but no results found.")
        except Exception:
            print(f"task_id {task_id!r} exists but unexpected error encountered.")

    # Only run
    logger.info(f"running simulation {sim_hash!r}")
    job = web.Job(simulation=sim, task_name=sim_hash, verbose=verbose)

    # Run simulation if results not found in local or server storage
    logger.info(f"sending Simulation {sim_hash!r} to tidy3d server.")
    return job.run(path=str(filepath))


def get_results(
    sim: td.Simulation,
    dirpath=PATH.results_tidy3d,
    overwrite: bool = True,
    verbose: bool = False,
) -> Awaitable[td.SimulationData]:
    """Return a List of SimulationData from a Simulation.

    Works with Pool of threads.
    Each thread can run in parallel and only becomes blocking when you ask
    for .result()

    Args:
        sims: List[Simulation]
        dirpath: to store results locally
        overwrite: overwrites the data even if path exists. Keep True.
        verbose: prints info messages and progressbars.

    .. code::
        import gdsfactory.simulation.tidy3d as gt

        component = gf.components.straight(length=3)
        sim = gt.get_simulation(component=component)
        sim_data = gt.get_results(sim) # threaded
        sim_data = sim_data.result() # waits for results

    """
    return _executor.submit(_get_results, sim, dirpath, overwrite, verbose)


def get_results_batch(
    sims: td.Simulation,
    dirpath=PATH.results_tidy3d,
    verbose: bool = True,
) -> td.BatchData:
    """Return a  a list of Simulation.

    Args:
        sims: List[Simulation]
        dirpath: to store results locally
        overwrite: overwrites the data even if path exists. Keep True.
        verbose: prints info messages and progressbars.

    .. code::
        import gdsfactory.simulation.tidy3d as gt

        component = gf.components.straight(length=3)
        sim = gt.get_simulation(component=component)
        sim_data = gt.get_results(sim) # threaded
        sim_data = sim_data.result() # waits for results

    """
    task_names = [get_sim_hash(sim) for sim in sims]
    batch = web.Batch(simulations=dict(zip(task_names, sims)), verbose=verbose)
    return batch.run(path_dir=dirpath)


if __name__ == "__main__":
    import gdsfactory.simulation.gtidy3d as gt

    component = gf.components.straight(length=3)
    sim = gt.get_simulation(component=component)
    sim_hash = get_sim_hash(sim)
    # hash_to_id = {d["taskName"]: d["task_id"] for d in web.get_tasks()}

    r = sim_data = get_results(sim=sim).result()

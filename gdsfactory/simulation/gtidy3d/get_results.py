"""Tidy3D """

import concurrent.futures
import hashlib
from typing import Awaitable

import tidy3d as td
from tidy3d import web

import gdsfactory as gf
from gdsfactory.config import PATH, logger
from gdsfactory.types import PathType

_executor = concurrent.futures.ThreadPoolExecutor()


def get_sim_hash(sim: td.Simulation) -> str:
    """Returns simulation hash as the unique ID."""
    return hashlib.md5(str(sim).encode()).hexdigest()


def _get_results(
    sim: td.Simulation,
    dirpath: PathType = PATH.results_tidy3d,
    overwrite: bool = False,
) -> td.SimulationData:
    """Return SimulationData results from simulation.

    Only submits simulation if results not found locally or remotely.
    First tries to load simulation results from disk.
    Then it tries to load them from the server storage.
    Finally, submits simulation to run remotely

    Args:
        sim: Simulation
        dirpath: to store results locally
        overwrite: overwrites the data even if path exists
    """
    task_name = sim_hash = get_sim_hash(sim)
    sim_path = dirpath / f"{sim_hash}.hdf5"
    logger.info(f"running simulation {sim_hash}")

    hash_to_id = {d["task_name"][:32]: d["task_id"] for d in web.get_tasks()}
    filepath = dirpath / f"{sim_hash}.hdf5"
    job = web.Job(simulation=sim, task_name=task_name)

    # Results in local storage
    if sim_path.exists():
        logger.info(f"{sim_path!r} found in local storage")
        sim_data = td.SimulationData.from_file(filepath)

    # Results in server storage
    elif sim_hash in hash_to_id:
        task_id = hash_to_id[sim_hash]
        web.monitor(task_id)
        sim_data = web.load(
            task_id=task_id, path=str(filepath), replace_existing=overwrite
        )

    # Run simulation if results not in local or server storage
    else:
        sim_data = job.run(path=filepath)
    return sim_data


def get_results(
    sim: td.Simulation,
    dirpath=PATH.results_tidy3d,
    overwrite: bool = True,
) -> Awaitable[td.SimulationData]:
    """Return a SimulationData from Simulation.

    Works with Pool of threads.
    Each thread can run in paralell and only becomes blocking when you ask
    for the result

    Args:
        sim: Simulation
        dirpath: to store results locally
        overwrite: overwrites the data even if path exists


    .. code::
        import gdsfactory.simulation.tidy3d as gt

        component = gf.components.straight(length=3)
        sim = gt.get_simulation(component=component)
        sim_data = gt.get_results(sim).result()

    """
    return _executor.submit(_get_results, sim, dirpath, overwrite)


if __name__ == "__main__":
    import gdsfactory.simulation.gtidy3d as gt

    component = gf.components.straight(length=3)
    sim = gt.get_simulation(component=component)
    r = sim_data = get_results(sim=sim).result()

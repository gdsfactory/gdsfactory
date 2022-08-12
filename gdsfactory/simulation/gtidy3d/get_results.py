"""Tidy3D."""

import concurrent.futures
import hashlib

import tidy3d as td
from tidy3d import web

import gdsfactory as gf
from gdsfactory.config import PATH
from gdsfactory.types import List, PathType

_executor = concurrent.futures.ThreadPoolExecutor()


def get_sim_hash(sim: td.Simulation) -> str:
    """Returns simulation hash as the unique ID."""
    return hashlib.md5(str(sim).encode()).hexdigest()


def _get_results(
    sims: List[td.Simulation],
    dirpath: PathType = PATH.results_tidy3d,
    overwrite: bool = False,
) -> web.BatchData:
    """Return SimulationData results from simulation.

    Only submits simulation if results not found locally or remotely.
    First tries to load simulation results from disk.
    Then it tries to load them from the server storage.
    Finally, submits simulation to run remotely

    Args:
        sim: tidy3d Simulation.
        dirpath: to store results locally.
        overwrite: overwrites the data even when path exists.

    """
    # task_name = sim_hash = get_sim_hash(sim)
    # sim_path = dirpath / f"{sim_hash}.hdf5"
    # logger.info(f"running simulation {sim_hash!r}")

    # hash_to_id = {d["task_name"][:32]: d["task_id"] for d in web.get_tasks()}
    # filepath = str(dirpath / f"{sim_hash}.hdf5")
    # job = web.Job(simulation=sim, task_name=task_name)

    # # Results in local storage
    # if sim_path.exists():
    #     task_id = hash_to_id[sim_hash]
    #     logger.info(f"{sim_path!r} for task_id {task_id!r} found in local storage")
    #     return td.SimulationData.from_file(filepath)

    # # Results in server storage
    # if sim_hash in hash_to_id:
    #     task_id = hash_to_id[sim_hash]
    #     web.monitor(task_id)

    #     try:
    #         return web.load(task_id=task_id, path=filepath, replace_existing=overwrite)
    #     except WebError:
    #         logger.info(f"task_id {task_id!r} exists but no results found.")
    #     except Exception:
    #         logger.info(f"task_id {task_id!r} exists but unexpected error encountered.")

    # # Run simulation if results not found in local or server storage
    # logger.info(f"sending task_name {task_name!r} to tidy3d server.")
    # return job.run(path=filepath)

    task_names = [get_sim_hash(sim) for sim in sims]
    batch = web.Batch(simulations=dict(zip(task_names, sims)))
    return batch.run(path_dir=dirpath)


def get_results(
    sims: List[td.Simulation],
    dirpath=PATH.results_tidy3d,
    overwrite: bool = True,
) -> web.BatchData:
    """Return a List of SimulationData from a list of Simulation.

    Works with Pool of threads.
    Each thread can run in parallel and only becomes blocking when you ask
    for .result()

    Args:
        sims: List[Simulation]
        dirpath: to store results locally
        overwrite: overwrites the data even if path exists. Keep True.


    .. code::
        import gdsfactory.simulation.tidy3d as gt

        component = gf.components.straight(length=3)
        sim = gt.get_simulation(component=component)
        sim_data = gt.get_results([sim])

    """
    return _get_results(sims, dirpath, overwrite)


if __name__ == "__main__":
    import gdsfactory.simulation.gtidy3d as gt

    component = gf.components.straight(length=3)
    sim = gt.get_simulation(component=component)
    r = sim_data = get_results(sim=sim).result()

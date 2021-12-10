"""Tidy3D """

import concurrent.futures
import hashlib
import json
import pathlib
from typing import Awaitable, Optional

import tidy3d as td
from tidy3d import web

from gdsfactory.config import PATH, logger

_executor = concurrent.futures.ThreadPoolExecutor()


def _export_simulation(
    sim: td.Simulation,
    task_name: Optional[str] = None,
    folder_name: str = "default",
    draft: bool = False,
) -> int:
    """Exports simulation to web and returns task_id.

    Args:
        sim: simulation object
        task_name:
        folder_name: Server folder to hold the task.
        draft: If ``True``, the project will be submitted but not run.
            It can then be visualized in the web UI and run from there when needed.

    """
    project = web.new_project(
        sim.export(), task_name=task_name, folder_name=folder_name, draft=draft
    )
    task_id = project["taskId"]
    logger.info(f"submitting {task_id}")
    return task_id


def get_sim_hash(sim: td.Simulation) -> str:
    """Returns simulation hash as the unique ID."""
    sim_str = json.dumps(sim.export())
    sim_hash = hashlib.md5(sim_str.encode()).hexdigest()
    return sim_hash


def load_results(
    sim: td.Simulation, target: pathlib.Path, task_id: Optional[str] = None
) -> td.Simulation:
    """Load results from HDF5 file. Returns a simulation that includes results.

    Args:
        sim: Simulation
        target: path
        task_id: Optional task Id
    """

    if task_id:
        web.monitor_project(task_id)
        src = "monitor_data.hdf5"
        web.download_results_file(task_id=task_id, src=src, target=str(target))

    sim.load_results(target)
    return sim


def run_simulation(sim: td.Simulation) -> Awaitable[td.Simulation]:
    """Returns a simulation with simulation results

    Only submits simulation if results not found locally or remotely.

    First tries to load simulation results from disk.
    Then it tries to load them from the server storage.
    Finally, only submits simulation if not found


    .. code::
        import gdsfactory.simulation.tidy3d as gm

        component = gf.components.straight(length=3)
        sim = gm.get_simulation(component=component)
        sim = run_simulation(sim).result()

    """
    td.logging_level("error")
    sim_hash = get_sim_hash(sim)
    sim_path = PATH.results_tidy3d / f"{sim_hash}.hdf5"
    logger.info(f"running simulation {sim_hash}")

    hash_to_id = {d["task_name"][:32]: d["task_id"] for d in web.get_last_projects()}
    target = PATH.results_tidy3d / f"{sim_hash}.hdf5"

    # Try from local storage
    if sim_path.exists():
        logger.info(f"{sim_path} found in local storage")
        sim = _executor.submit(load_results, sim, target)

    # Try from server storage
    elif sim_hash in hash_to_id:
        task_id = hash_to_id[sim_hash]
        sim = _executor.submit(load_results, sim, target, task_id)

    # Only submit if simulation not found
    else:
        task_id = _export_simulation(sim=sim, task_name=sim_hash)
        sim = _executor.submit(load_results, sim, target, task_id)
    return sim


if __name__ == "__main__":
    import gdsfactory as gf
    import gdsfactory.simulation.tidy3d as gm

    # for length in range(12, 13):
    #     component = gf.components.straight(length=length)
    #     sim = gm.get_simulation(component=component)
    #     s = run_simulation(sim, wait_to_complete=False)

    component = gf.components.straight(length=3)
    s = gm.get_simulation(component=component)
    r = run_simulation(s).result()

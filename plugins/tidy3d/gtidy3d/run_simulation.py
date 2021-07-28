from typing import Optional
import hashlib
import json
import tidy3d as td
from tidy3d import web

from gtidy3d.config import logger, PATH


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


def run_simulation(sim: td.Simulation, wait_to_complete: bool = True) -> td.Simulation:
    """Runs simulation.

    Only creates and submits simulation if results not found locally or remotely.
    """
    sim_hash = get_sim_hash(sim)
    sim_path = PATH.results / f"{sim_hash}.hdf5"
    logger.info(f"running simulation {sim_hash}")

    hash_to_id = {d["task_name"][:32]: d["task_id"] for d in web.get_last_projects()}

    # Try from local storage
    if sim_path.exists():
        logger.info(f"{sim_path} exists, loading it directly")
        target = PATH.results / f"{sim_hash}.hdf5"
        sim.load_results(target)
        # sim = td.Simulation.import_json("out/simulation.json")

    # Try from server storage
    elif sim_hash in hash_to_id:
        task_id = hash_to_id[sim_hash]
        logger.info(f"downloading task_id = {task_id} with hash {sim_hash}")

        src = "tidy3d.log"
        target = PATH.results / f"{sim_hash}.log"
        web.download_results_file(task_id=task_id, src=src, target=str(target))

        src = "monitor_data.hdf5"
        target = PATH.results / f"{sim_hash}.hdf5"
        web.download_results_file(task_id=task_id, src=src, target=str(target))

    # Only submit if simulation not found
    else:
        task_id = _export_simulation(sim=sim, task_name=sim_hash)

        if wait_to_complete:
            web.monitor_project(task_id)
            src = "tidy3d.log"
            target = PATH.results / f"{sim_hash}.log"
            web.download_results_file(task_id=task_id, src=src, target=str(target))

            src = "monitor_data.hdf5"
            target = PATH.results / f"{sim_hash}.hdf5"
            web.download_results_file(task_id=task_id, src=src, target=str(target))
            sim.load_results(target)
        else:
            logger.info(
                f"Adding simulation with task_id = {task_id} to the simulation queue"
            )
    return sim


if __name__ == "__main__":
    import pp
    import gtidy3d as gm

    for length in [1, 5]:
        component = pp.components.straight(length=length)
        sim = gm.get_simulation(component=component)
        run_simulation(sim, wait_to_complete=False)

    # component = pp.components.straight(length=3)
    # sim = gm.get_simulation(component=component)
    # sim = run_simulation(sim)

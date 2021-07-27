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
    """Exports simulation to web and returns taskId.

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
    taskId = project["taskId"]
    logger.info(f"submitting {taskId}")
    return taskId


def get_sim_hash(sim: td.Simulation) -> str:
    """Returns simulation hash as the unique ID."""
    sim_str = json.dumps(sim.export())
    sim_hash = hashlib.md5(sim_str.encode()).hexdigest()
    return sim_hash


def run_simulation(sim: td.Simulation) -> td.Simulation:
    """If simulation exists, runs simulation.

    Only creates and submitts new project if simulation results are found locally.
    """
    sim_hash = get_sim_hash(sim)
    sim_path = PATH.results / f"{sim_hash}.hdf5"
    logger.info(f"running simulation {sim_hash}")

    if sim_path.exists():
        logger.info(f"{sim_path} exists, loading it directly")
        target = PATH.results / f"{sim_hash}.hdf5"
        sim.load_results(target)
        # sim = td.Simulation.import_json("out/simulation.json")

    else:
        taskId = _export_simulation(sim=sim, task_name=sim_hash)
        web.monitor_project(taskId)

        src = "tidy3d.log"
        target = PATH.results / f"{sim_hash}.log"
        web.download_results_file(task_id=taskId, src=src, target=str(target))

        src = "monitor_data.hdf5"
        target = PATH.results / f"{sim_hash}.hdf5"
        web.download_results_file(task_id=taskId, src=src, target=str(target))

        sim.load_results(target)
    return sim


if __name__ == "__main__":
    import pp
    import gtidy3d as gm

    component = pp.components.straight(length=3)
    sim = gm.get_simulation(component=component)
    sim = run_simulation(sim)

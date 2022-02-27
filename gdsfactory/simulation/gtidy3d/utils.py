"""gdsfactory tidy3d plugin utils"""

from functools import partial

from tidy3d import web

from gdsfactory.config import logger


def delete_tasks(status: str = "error") -> None:
    """Deletes all tasks with a particular status.

    Args:
        status: error, diverged, preprocess
    """

    for task in web.get_tasks():
        # print(task["status"], task["task_name"], task["task_id"])
        if task["status"] == status:
            task_id = task["task_id"]
            logger.info(f"deleted failed task_id {task_id}")
            web.delete(task_id)


delete_tasks_failed = partial(delete_tasks, status="error")
delete_tasks_preprocess = partial(delete_tasks, status="preprocess")


if __name__ == "__main__":
    delete_tasks_failed()

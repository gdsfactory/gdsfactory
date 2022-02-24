"""gdsfactory tidy3d plugin utils"""

from tidy3d import web

from gdsfactory.config import logger


def delete_failed_tasks():
    """Deletes all failed tasks"""
    for task in web.get_tasks():
        if task["status"] == "error":
            task_id = task["task_id"]
            logger.info(f"deleted failed task_id {task_id}")
            web.delete(task_id)

from __future__ import annotations

import logging
import pathlib
import tempfile
import time
from unittest.mock import MagicMock

from gdsfactory.watch import FileWatcher

wait_time = 0.2


def test_file_watcher() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_logger = MagicMock(spec=logging.Logger)

        watcher = FileWatcher(
            path=tmp_dir, run_main=False, run_cells=True, logger=mock_logger
        )
        watcher.start()
        time.sleep(wait_time)

        py_path = pathlib.Path(tmp_dir) / "test.py"
        py_content = """
import gdsfactory as gf

def straight():
    return gf.components.straight(length=10, width=0.5)
"""
        py_path.write_text(py_content)
        time.sleep(wait_time)
        mock_logger.info.assert_called()

        yaml_path = pathlib.Path(tmp_dir) / "test.pic.yml"
        yaml_content = """
name: test_component
instances:
    mmi:
        component: mmi1x2
        settings:
            width_mmi: 4.5
            length_mmi: 10
"""
        yaml_path.write_text(yaml_content)
        time.sleep(wait_time)
        mock_logger.info.assert_called()

        py_path.write_text(py_content + "\n# Modified")
        time.sleep(wait_time)
        mock_logger.info.assert_called()

        yaml_path.write_text(yaml_content + "\n# Modified")
        time.sleep(wait_time)
        mock_logger.info.assert_called()

        yaml_path.unlink()
        time.sleep(wait_time)
        mock_logger.info.assert_called()

        watcher.stop()


def test_file_watcher_ignored_files() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_logger = MagicMock(spec=logging.Logger)

        watcher = FileWatcher(
            path=tmp_dir, run_main=False, run_cells=True, logger=mock_logger
        )
        watcher.start()

        txt_path = pathlib.Path(tmp_dir) / "test.txt"
        txt_path.write_text("Some text")
        time.sleep(wait_time)

        mock_logger.info.assert_not_called()

        watcher.stop()


if __name__ == "__main__":
    test_file_watcher()
    test_file_watcher_ignored_files()

from __future__ import annotations

import logging
import pathlib
import tempfile
import time
from unittest.mock import MagicMock

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)

import gdsfactory as gf
from gdsfactory.pdk import get_active_pdk
from gdsfactory.watch import FileWatcher, watch


def _wait_for_log_message(
    mock_logger: MagicMock, message: str, timeout: float = 5
) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        for call in mock_logger.info.call_args_list:
            print(f"call: {call}")
            if message.lower() in str(call).lower():
                return True
        time.sleep(0.1)
    return False


def test_file_watcher() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_logger = MagicMock(spec=logging.Logger)

        watcher = FileWatcher(
            path=tmp_dir, run_main=False, run_cells=True, logger=mock_logger
        )
        watcher.start()
        time.sleep(0.5)

        py_path = pathlib.Path(tmp_dir) / "test.py"
        py_content = """
import gdsfactory as gf

def straight():
    return gf.components.straight(length=10, width=0.5)
"""
        py_path.write_text(py_content)
        assert _wait_for_log_message(mock_logger, "Created")

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
        assert _wait_for_log_message(mock_logger, "Created")

        py_path.write_text(py_content + "\n# Modified")
        assert _wait_for_log_message(mock_logger, "Modified")

        new_yaml_path = pathlib.Path(tmp_dir) / "new_test.pic.yml"
        yaml_path.rename(new_yaml_path)
        res = _wait_for_log_message(mock_logger, "Moved")
        print(f"res: {res}")
        assert res
        new_yaml_path.unlink()
        assert _wait_for_log_message(mock_logger, "Deleted")

        watcher.stop()


def test_file_watcher_ignored_files() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_logger = MagicMock(spec=logging.Logger)

        watcher = FileWatcher(
            path=tmp_dir, run_main=False, run_cells=True, logger=mock_logger
        )
        watcher.start()
        time.sleep(1)

        txt_path = pathlib.Path(tmp_dir) / "test.txt"
        txt_path.write_text("Some text")

        mock_logger.info.assert_not_called()

        watcher.stop()


def test_update_cell_invalid_yaml() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        watcher = FileWatcher(path=tmp_dir)

        pdk = gf.get_active_pdk()
        cells_before = list(pdk.cells.keys())
        yaml_path = pathlib.Path(tmp_dir) / f"{cells_before[0]}.pic.yml"
        yaml_content = """
invalid:
  - yaml: content
    indentation
"""
        yaml_path.write_text(yaml_content)

        watcher.update_cell(yaml_path)
        cells_after = list(pdk.cells.keys())
        assert set(cells_before) == set(cells_after)


def test_file_watcher_run_loop() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_logger = MagicMock(spec=logging.Logger)
        watcher = FileWatcher(path=tmp_dir, logger=mock_logger)

        mock_observer = MagicMock()
        mock_observer.is_alive.side_effect = [False, True, True]
        watcher.observer = mock_observer

        watcher.start()
        time.sleep(0.1)

        assert mock_observer.start.call_count == 1

        time.sleep(1.1)
        assert mock_observer.start.call_count == 1

        watcher.stop()

        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()


def test_on_moved() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_logger = MagicMock(spec=logging.Logger)
        watcher = FileWatcher(path=tmp_dir, logger=mock_logger)
        watcher.start()
        time.sleep(0.1)

        yaml_path = pathlib.Path(tmp_dir) / "file.pic.yml"
        yaml_path.write_text("name: test\ninstances: {}")

        create_event = FileCreatedEvent(src_path=str(yaml_path))
        watcher.on_created(create_event)

        new_path = pathlib.Path(tmp_dir) / "moved_file.pic.yml"
        yaml_path.rename(new_path)
        event = FileMovedEvent(src_path=str(yaml_path), dest_path=str(new_path))
        watcher.on_moved(event)

        assert _wait_for_log_message(mock_logger, "Moved")

        watcher.stop()


def test_on_created() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_logger = MagicMock(spec=logging.Logger)
        watcher = FileWatcher(path=tmp_dir, logger=mock_logger)

        yaml_path = pathlib.Path(tmp_dir) / "file.pic.yml"
        yaml_path.write_text("name: test\ninstances: {}")
        event = FileCreatedEvent(src_path=str(yaml_path))
        watcher.on_created(event)
        mock_logger.info.assert_called_with("Created %s: %s", "file", str(yaml_path))

        py_path = pathlib.Path(tmp_dir) / "file.py"
        py_path.write_text("# test file")
        event = FileCreatedEvent(src_path=str(py_path))
        watcher.on_created(event)
        mock_logger.info.assert_called_with("Created %s: %s", "file", str(py_path))

        other_path = pathlib.Path(tmp_dir) / "file.txt"
        event = FileCreatedEvent(src_path=str(other_path))
        watcher.on_created(event)
        assert mock_logger.info.call_count == 2


def test_on_deleted() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_logger = MagicMock(spec=logging.Logger)
        watcher = FileWatcher(path=tmp_dir, logger=mock_logger)

        pdk = get_active_pdk().model_copy(deep=True)
        pdk.activate()

        yaml_path = pathlib.Path(tmp_dir) / "yaml_component.pic.yml"
        yaml_content = """
name: yaml_component
"""
        yaml_path.write_text(yaml_content)

        create_event_yaml = FileCreatedEvent(src_path=str(yaml_path))
        watcher.on_created(create_event_yaml)
        time.sleep(0.5)

        print(list(pdk.cells.keys()))

        event = FileDeletedEvent(src_path=str(yaml_path))
        watcher.on_deleted(event)
        assert _wait_for_log_message(mock_logger, "Deleted")
        assert _wait_for_log_message(mock_logger, "yaml_component")


def test_on_modified() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_logger = MagicMock(spec=logging.Logger)
        watcher = FileWatcher(path=tmp_dir, logger=mock_logger)

        yaml_path = pathlib.Path(tmp_dir) / "file.pic.yml"
        yaml_path.write_text("name: test\ninstances: {}")
        event = FileModifiedEvent(src_path=str(yaml_path))
        watcher.on_modified(event)
        mock_logger.info.assert_called_with("Modified %s: %s", "file", str(yaml_path))

        py_path = pathlib.Path(tmp_dir) / "file.py"
        py_path.write_text("# test file")
        event = FileModifiedEvent(src_path=str(py_path))
        watcher.on_modified(event)
        mock_logger.info.assert_called_with("Modified %s: %s", "file", str(py_path))

        mock_logger.info.reset_mock()
        other_path = pathlib.Path(tmp_dir) / "file.txt"
        event = FileModifiedEvent(src_path=str(other_path))
        watcher.on_modified(event)
        assert mock_logger.info.call_count == 0

        str_path = str(pathlib.Path(tmp_dir) / "file.py")
        event = FileModifiedEvent(src_path=str_path)
        watcher.on_modified(event)
        mock_logger.info.assert_called_with("Modified %s: %s", "file", str_path)


def test_get_component() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_logger = MagicMock(spec=logging.Logger)
        watcher = FileWatcher(path=tmp_dir, logger=mock_logger)

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
        component = watcher.get_component(yaml_path)
        assert component is not None

        py_path = pathlib.Path(tmp_dir) / "test.py"
        py_content = """
import gdsfactory as gf

def straight():
    return gf.components.straight(length=10, width=0.5)
"""
        py_path.write_text(py_content)
        component = watcher.get_component(py_path)
        assert component is None

        watcher.run_main = True
        component = watcher.get_component(py_path)
        assert component is None

        watcher.run_cells = False
        component = watcher.get_component(py_path)
        assert component is None

        nonexistent = pathlib.Path(tmp_dir) / "nonexistent.pic.yml"
        component = watcher.get_component(nonexistent)
        assert component is None

        other_path = pathlib.Path(tmp_dir) / "test.txt"
        other_path.write_text("some content")
        component = watcher.get_component(other_path)
        assert component is None


def test_watch() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_logger = MagicMock(spec=logging.Logger)

        watch(
            path=tmp_dir,
            pdk=None,
            run_main=True,
            run_cells=True,
            pre_run=False,
            logger=mock_logger,
            run_embed=False,
        )

        mock_logger.info.assert_called()

        pdk = get_active_pdk().model_copy(deep=True)

        watch(
            path=tmp_dir,
            pdk=pdk,
            pre_run=True,
            logger=mock_logger,
            run_embed=False,
        )

        py_path = pathlib.Path(tmp_dir) / "test.py"
        py_path.write_text("import gdsfactory as gf")

        watch(path=tmp_dir, pre_run=True, logger=mock_logger, run_embed=False)


if __name__ == "__main__":
    test_on_moved()

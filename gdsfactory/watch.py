"""FileWatcher based on watchdog. Looks for changes in files with .pic.yml extension."""

from __future__ import annotations

import logging
import os
import pathlib
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from typing import TypeAlias

import kfactory as kf
from IPython.terminal.embed import embed
from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from gdsfactory.component import Component
from gdsfactory.config import cwd
from gdsfactory.pdk import Pdk, get_active_pdk
from gdsfactory.read.from_yaml_template import cell_from_yaml_template
from gdsfactory.typings import ComponentFactory, ComponentSpec, PathType

_MovedEvent: TypeAlias = DirMovedEvent | FileMovedEvent
_CreatedEvent: TypeAlias = DirCreatedEvent | FileCreatedEvent
_DeletedEvent: TypeAlias = DirDeletedEvent | FileDeletedEvent
_ModifiedEvent: TypeAlias = DirModifiedEvent | FileModifiedEvent


class FileWatcher(FileSystemEventHandler):
    """Captures *.py or *.pic.yml file change events."""

    def __init__(
        self,
        path: str,
        run_main: bool = False,
        run_cells: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the YAML event handler.

        Args:
            path: the path to the directory to watch.
            run_main: if True, will execute the main function of the file.
            run_cells: if True, will execute the cells of the file.
            logger: the logger to use.
        """
        super().__init__()

        self.logger = logger or logging.root
        self.run_cells = run_cells
        self.run_main = run_main

        pdk = get_active_pdk()
        pdk.register_cells_yaml(dirpath=path, update=True)

        self.observer = Observer()
        self.path = path
        self.stopping = threading.Event()

    def start(self) -> None:
        self.observer.schedule(self, self.path, recursive=True)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self) -> None:
        while not self.stopping.is_set():
            if not self.observer.is_alive():
                self.observer.start()
            time.sleep(1)
        self.observer.stop()
        self.observer.join()

    def stop(self) -> None:
        self.stopping.set()
        self.thread.join()

    def update_cell(self, src_path: PathType, update: bool = False) -> ComponentFactory:
        """Parses a YAML file to a cell function and registers into active pdk.

        Args:
            src_path: the path to the file
            update: if True, will update an existing cell function of the same name without raising an error
        Returns:
            The cell function parsed from the yaml file.

        """
        pdk = get_active_pdk()
        print(f"Active PDK: {pdk.name!r}")
        filepath = pathlib.Path(src_path)
        cell_name = filepath.stem.split(".")[0]
        function = cell_from_yaml_template(filepath, name=cell_name)
        try:
            pdk.register_cells_yaml(update=update, **{cell_name: function})  # type: ignore[arg-type]
        except ValueError as e:
            print(e)
        return function

    def _get_path(self, path: str | bytes) -> str:
        return path.decode("utf-8") if isinstance(path, bytes) else path

    def on_moved(self, event: _MovedEvent) -> None:
        super().on_moved(event)

        what = "directory" if event.is_directory else "file"
        dest_path = self._get_path(event.dest_path)

        if what == "file" and dest_path.endswith(".pic.yml"):
            self.logger.info("Moved %s: %s", what, dest_path)
            self.update_cell(dest_path)
            self.get_component(dest_path)

    def on_created(self, event: _CreatedEvent) -> None:
        super().on_created(event)

        what = "directory" if event.is_directory else "file"
        src_path = self._get_path(event.src_path)
        if (what == "file" and src_path.endswith(".pic.yml")) or src_path.endswith(
            ".py"
        ):
            self.logger.info("Created %s: %s", what, src_path)
            self.get_component(src_path)

    def on_deleted(self, event: _DeletedEvent) -> None:
        super().on_deleted(event)

        what = "directory" if event.is_directory else "file"
        src_path = self._get_path(event.src_path)

        if what == "file" and src_path.endswith(".pic.yml"):
            self.logger.info("Deleted %s: %s", what, event.src_path)
            pdk = get_active_pdk()
            filepath = pathlib.Path(src_path)
            cell_name = filepath.stem.split(".")[0]
            pdk.remove_cell(cell_name)

    def on_modified(self, event: _ModifiedEvent) -> None:
        super().on_modified(event)

        # Determine file type
        what = "directory" if event.is_directory else "file"
        if not isinstance(event.src_path, str):
            src_path = event.src_path.decode("utf-8")
        else:
            src_path = event.src_path

        # Check if the file matches the extensions we care about
        if what == "file" and (
            src_path.endswith(".pic.yml") or src_path.endswith(".py")
        ):
            self.logger.info("Modified %s: %s", what, src_path)
            self.get_component(src_path)
        else:
            print(f"Ignored {what}: {src_path}")

    def get_component(self, filepath: PathType) -> Component | None:
        import git
        import git.repo as gr

        from gdsfactory.get_factories import get_cells_from_dict

        try:
            repo = gr.Repo(".", search_parent_directories=True)
            dirpath = repo.working_tree_dir
        except git.InvalidGitRepositoryError:
            dirpath = cwd
        if dirpath is None:
            dirpath = cwd
        try:
            filepath = pathlib.Path(filepath)
            dirpath = pathlib.Path(dirpath) / "build/gds"
            dirpath.mkdir(parents=True, exist_ok=True)

            if filepath.exists():
                if str(filepath).endswith(".pic.yml"):
                    return self.get_component_yaml(filepath, dirpath)
                elif str(filepath).endswith(".py"):
                    context = dict(locals(), **globals())
                    if self.run_main:
                        context.update(__name__="__main__")

                    # Read the content of the file and execute it within the updated context
                    exec(filepath.read_text(), context, context)

                    if self.run_cells:
                        cells = get_cells_from_dict(context)
                        # Process each cell and write it to a GDS file
                        for name, cell in cells.items():
                            c = cell()
                            gdspath = dirpath / f"{name}.gds"
                            c.write_gds(gdspath)
                            kf.show(gdspath)

                else:
                    print(f"Changed file {filepath} ignored (not .pic.yml or .py)")

        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            print(e)
        return None

    def get_component_yaml(self, filepath: PathType, dirpath: PathType) -> Component:
        """Parses a YAML file to a cell function and registers into active pdk."""
        cell_func = self.update_cell(filepath, update=True)
        filepath_path = pathlib.Path(filepath)
        c = cell_func()
        gdspath = pathlib.Path(dirpath) / str(
            filepath_path.relative_to(self.path)
        ).replace(".pic.yml", ".gds")
        c.write_gds(gdspath)
        kf.show(gdspath)
        return c


def watch(
    path: PathType | None = cwd,
    pdk: Pdk | str | None = None,
    run_main: bool = True,
    run_cells: bool = True,
    pre_run: bool = False,
    logger: logging.Logger | None = None,
    run_embed: bool = True,
) -> None:
    """Starts the file watcher.

    Args:
        path: the path to the directory to watch.
        pdk: the name of the PDK to use.
        run_main: if True, will execute the main function of the file.
        run_cells: if True, will execute the cells of the file.
        pre_run: build all cells on startup
        logger: the logger to use.
        run_embed: if True, will run the embed function.
    """
    path = str(path)
    logger = logger or logging.root
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if pdk:
        if isinstance(pdk, str):
            get_active_pdk(name=pdk)
        else:
            pdk.activate()
    pdk_name = get_active_pdk().name if pdk else None

    print(f"Watching {path=}, {pdk_name=} {run_main=}, {run_cells=}, {pre_run=}")
    watcher = FileWatcher(
        path=path, run_main=run_main, run_cells=run_cells, logger=logger
    )
    watcher.start()
    if pre_run:
        for root, _, fns in os.walk(path):
            for fn in fns:
                path = os.path.join(root, fn)
                if path.endswith(".py") or path.endswith(".pic.yml"):
                    event = SimpleNamespace(is_directory=False, src_path=path)
                    watcher.on_created(event)  # type: ignore

    logger.info(
        f"File watcher looking for changes in *.py and *.pic.yml files in {path!r}. Stop with Ctrl+C"
    )
    if run_embed:
        embed()
    watcher.stop()


def show(component: ComponentSpec) -> None:
    """Shows a component in klayout."""
    import gdsfactory as gf

    c = gf.get_component(component)
    c.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    watch(path)

import logging
import sys
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import gdsfactory as gf
from gdsfactory.config import cwd


class YamlEventHandler(FileSystemEventHandler):
    """Logs all the events captured."""

    def __init__(self, logger=None):
        super().__init__()

        self.logger = logger or logging.root

    def on_moved(self, event):
        super().on_moved(event)

        what = "directory" if event.is_directory else "file"
        self.logger.info(
            "Moved %s: from %s to %s", what, event.src_path, event.dest_path
        )

    def on_created(self, event):
        super().on_created(event)

        what = "directory" if event.is_directory else "file"
        if what == "file" and event.src_path.endswith(".pic.yml"):
            self.logger.info("Created %s: %s", what, event.src_path)
            c = gf.read.from_yaml(event.src_path)
            c.show()

    def on_deleted(self, event):
        super().on_deleted(event)

        what = "directory" if event.is_directory else "file"
        self.logger.info("Deleted %s: %s", what, event.src_path)

    def on_modified(self, event):
        super().on_modified(event)

        what = "directory" if event.is_directory else "file"
        if what == "file" and event.src_path.endswith(".pic.yml"):
            self.logger.info("Modified %s: %s", what, event.src_path)
            try:
                c = gf.read.from_yaml(event.src_path)
                c.show()
            except Exception as e:
                print(e)


def watch(path=str(cwd)) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    event_handler = YamlEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    watch(path)

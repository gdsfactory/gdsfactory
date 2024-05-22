"""Gdsfactory configuration."""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import traceback
from enum import Enum, auto
from itertools import takewhile
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING, Any

import loguru
from dotenv import find_dotenv
from kfactory.conf import config
from loguru import logger as logger
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    pass

__version__ = "7.8.5"
PathType = str | pathlib.Path

home = pathlib.Path.home()
cwd = pathlib.Path.cwd()
module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent
home_path = pathlib.Path.home() / ".gdsfactory"
diff_path = repo_path / "gds_diff"
logpath = home_path / "log.log"
dotenv_path = find_dotenv(usecwd=True)

GDSDIR_TEMP = pathlib.Path(tempfile.TemporaryDirectory().name).parent / "gdsfactory"

plugins = [
    "gplugins",
    "ray",
    "femwell",
    "devsim",
    "tidy3d",
    "meep",
    "meow",
    "lumapi",
    "sax",
]
pdks = [
    "aim",
    "amf",
    "ct",
    "gf180",
    "gf45",
    "hhi",
    "imec",
    "sky130",
    "sph",
    "tj",
    "ubcpdk",
    "gvtt",
]


class LogLevel(str, Enum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFilter(BaseModel):
    """Filter certain messages by log level or regex.

    Filtered messages are not evaluated and discarded.
    """

    level: LogLevel = LogLevel.INFO
    regex: str | None = None

    def __call__(self, record: loguru.Record) -> bool:
        """Loguru needs the filter to be callable."""
        levelno = logger.level(self.level).no
        if self.regex is None:
            return record["level"].no >= levelno
        else:
            return record["level"].no >= levelno and not bool(
                re.search(self.regex, record["message"])
            )


class ErrorType(Enum):
    ERROR = auto()
    WARNING = auto()
    IGNORE = auto()


def print_version_plugins() -> None:
    """Print gdsfactory plugin versions and paths."""
    table = Table(title="Modules")
    table.add_column("Package", justify="right", style="cyan", no_wrap=True)
    table.add_column("version", style="magenta")
    table.add_column("Path", justify="right", style="green")

    table.add_row("python", sys.version, str(sys.executable))
    table.add_row("gdsfactory", __version__, str(module_path))

    for plugin in plugins:
        try:
            m = importlib.import_module(plugin)
            try:
                table.add_row(plugin, str(m.__version__), str(m.__path__))
            except AttributeError:
                table.add_row(plugin, "", "")
        except ImportError:
            table.add_row(plugin, "not installed", "")

    console = Console()
    console.print(table)


def print_version_plugins_raw() -> None:
    """Print gdsfactory plugin versions and paths."""
    print("python", sys.version)
    print("gdsfactory", __version__)

    for plugin in plugins:
        try:
            m = importlib.import_module(plugin)
            try:
                print(plugin, m.__version__)
            except AttributeError:
                print(plugin)
        except ImportError:
            print(plugin, "not installed", "")


def print_version_pdks() -> None:
    """Print gdsfactory PDK versions and paths."""
    table = Table(title="PDKs")
    table.add_column("Package", justify="right", style="cyan", no_wrap=True)
    table.add_column("version", style="magenta")
    table.add_column("Path", justify="right", style="green")

    for pdk in pdks:
        try:
            m = importlib.import_module(pdk)
            try:
                table.add_row(pdk, str(m.__version__), str(m.__path__))
            except AttributeError:
                table.add_row(pdk, "", "")
        except ImportError:
            table.add_row(pdk, "not installed", "")

    console = Console()
    console.print(table)


def get_number_of_cores() -> int:
    """Get number of cores/threads available.

    On (most) linux we can get it through the scheduling affinity. Otherwise,
    fall back to the multiprocessing cpu count.
    """
    try:
        threads = len(os.sched_getaffinity(0))
    except AttributeError:
        import multiprocessing

        threads = multiprocessing.cpu_count()
    return threads


def tracing_formatter(record: loguru.Record) -> str:
    """Traceback filtering.

    Filter out frames coming from Loguru internals.
    """
    frames = takewhile(
        lambda f: "/loguru/" not in f.filename, traceback.extract_stack()
    )
    stack = " > ".join(f"{f.filename}:{f.name}:{f.lineno}" for f in frames)
    record["extra"]["stack"] = stack

    if record["extra"].get("with_backtrace", False):
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level>"
            " | <cyan>{extra[stack]}</cyan> - <level>{message}</level>\n{exception}"
        )

    else:
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}"
            "</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
            " - <level>{message}</level>\n{exception}"
        )


CONF = config
CONF.difftest_ignore_label_differences = False
CONF.difftest_ignore_sliver_differences = False
CONF.difftest_ignore_cell_name_differences = True
CONF.bend_radius_error_type = ErrorType.ERROR
CONF.layer_error_path = (1000, 0)
CONF.pdk = None


class Paths:
    module = module_path
    repo = repo_path
    results_tidy3d = home / ".tidy3d"
    generic_tech = module / "generic_tech"
    klayout = generic_tech / "klayout"
    klayout_tech = klayout
    klayout_lyp = klayout_tech / "layers.lyp"
    klayout_yaml = generic_tech / "layer_views.yaml"
    schema_netlist = repo_path / "tests" / "schemas" / "netlist.json"
    netlists = module_path / "samples" / "netlists"
    gdsdir = repo_path / "tests" / "gds"
    gdslib = home / ".gdsfactory"
    modes = gdslib / "modes"
    sparameters = gdslib / "sp"
    capacitance = gdslib / "capacitance"
    interconnect = gdslib / "interconnect"
    optimiser = repo_path / "tune"
    notebooks = repo_path / "docs" / "notebooks"
    test_data = repo / "test-data-gds"
    gds_ref = test_data / "gds"
    gds_run = GDSDIR_TEMP / "gds_run"
    gds_diff = GDSDIR_TEMP / "gds_diff"
    cwd = cwd
    sparameters_repo = test_data / "sp"  # repo with some demo sparameters
    fonts = module / "components" / "fonts"
    font_ocr = fonts / "OCR-A.ttf"


PATH = Paths()
sparameters_path = PATH.sparameters


def rich_output() -> None:
    """Enables rich output."""
    from rich import pretty

    pretty.install()


def complex_encoder(z):
    if isinstance(z, pathlib.Path):
        return str(z)
    elif callable(z):
        return str(z.__name__)
    else:
        type_name = type(z)
        raise TypeError(f"Object {z} of type {type_name} is not serializable")


def write_config(config: Any, json_out_path: Path) -> None:
    """Write config to a JSON file."""
    with open(json_out_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=complex_encoder)


def print_config(key: str | None = None) -> None:
    """Prints a key for the config or all the keys."""
    if key:
        if CONF.get(key):
            print(CONF[key])
        else:
            print(f"{key!r} key not found in {CONF.keys()}")
    else:
        pprint(CONF)


def call_if_func(f: Any, **kwargs) -> Any:
    """Calls function if it's a function Useful to create objects from.

    functions if it's an object it just returns the object.
    """
    return f(**kwargs) if callable(f) else f


def get_git_hash():
    """Returns repository git hash."""
    try:
        with open(os.devnull, "w") as shutup:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=shutup)
                .decode("utf-8")
                .strip("\n")
            )
    except subprocess.CalledProcessError:
        return "not_a_git_repo"


if __name__ == "__main__":
    pass
    # print(CONF.pdk)
    # print_version_plugins()
    # print_version_pdks()

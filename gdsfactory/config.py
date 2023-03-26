"""Gdsfactory loads configuration from 3 files, high priority overwrites low.

priority:

1. A config.yml found in the current working directory (highest priority)
2. ~/.gdsfactory/config.yml specific for the machine
3. the yamlpath_default in gdsfactory.technology.yml (lowest priority)

You can access the CONF dictionary with `print_config`

PATH has all your computer specific paths that we do not care to store

"""

from __future__ import annotations

import warnings
import sys
import io
import json
import os
import pathlib
import subprocess
from pathlib import Path
from pprint import pprint
from typing import Any, Iterable, Optional, Union
import importlib

from loguru import logger
import omegaconf
from omegaconf import OmegaConf

from rich.console import Console
from rich.table import Table

__version__ = "6.71.0"
PathType = Union[str, pathlib.Path]

home = pathlib.Path.home()
cwd = pathlib.Path.cwd()
module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent
home_path = pathlib.Path.home() / ".gdsfactory"
diff_path = repo_path / "gds_diff"
logpath = home_path / "log.log"

yamlpath_cwd = cwd / "config.yml"
yamlpath_default = module_path / "config.yml"
yamlpath_home = home_path / "config.yml"

MAX_NAME_LENGTH = 32

logger.remove()
logger.add(sink=sys.stderr, level="WARNING")

showwarning_ = warnings.showwarning


def showwarning(message, *args, **kwargs):
    logger.warning(message)
    showwarning_(message, *args, **kwargs)


warnings.showwarning = showwarning

plugins = ["ray", "femwell", "devsim", "tidy3d", "meep", "meow", "lumapi", "sax"]
pdks = ["gf45", "tj", "imec", "amf", "sky130", "ubcpdk", "aim", "ct"]


def print_version():
    """Print gdsfactory plugin versions and paths."""
    table = Table(title="Modules")
    table.add_column("Package", justify="right", style="cyan", no_wrap=True)
    table.add_column("version", style="magenta")
    table.add_column("Path", justify="right", style="green")

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


def print_version_raw():
    """Print gdsfactory plugin versions and paths."""
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


def print_version_pdks():
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


default_config = io.StringIO(
    """
plotter: matplotlib
sparameters_path: ${oc.env:HOME}/.gdsfactory/sparameters/generic
show_ports: True
"""
)


def set_log_level(level: str, sink=sys.stderr) -> None:
    """Sets log level for gdsfactory.

    Args:
        level: ["DEBUG", "INFO", "WARNING", "ERROR"]
        sink: defaults to standard error.
    """
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    if level not in log_levels:
        raise ValueError(f"{level!r} not a valid log level {log_levels}")
    logger.remove()
    logger.add(sink=sink, level=level)


class Paths:
    module = module_path
    repo = repo_path
    results_tidy3d = home / ".tidy3d"
    generic_tech = module / "generic_tech"
    klayout = generic_tech / "klayout"
    klayout_tech = klayout / "tech"
    klayout_lyp = klayout_tech / "layers.lyp"
    klayout_yaml = generic_tech / "layer_views.yaml"
    schema_netlist = repo_path / "tests" / "schemas" / "netlist.json"
    netlists = module_path / "samples" / "netlists"
    gdsdir = repo_path / "tests" / "gds"
    gdslib = repo_path / "gdslib"
    modes = gdslib / "modes"
    gdsdiff = gdslib / "gds"
    sparameters = gdslib / "sp"
    interconnect = gdslib / "interconnect"
    optimiser = repo_path / "tune"
    notebooks = repo_path / "docs" / "notebooks"


def read_config(
    yamlpaths: Iterable[PathType] = (yamlpath_default, yamlpath_home, yamlpath_cwd),
) -> omegaconf.DictConfig:
    config = OmegaConf.load(default_config)
    for yamlpath in set(yamlpaths):
        yamlpath = pathlib.Path(yamlpath)
        if os.access(yamlpath, os.R_OK) and yamlpath.exists():
            logger.info(f"loading tech config from {yamlpath}")
            config_new = OmegaConf.load(yamlpath)
            config = OmegaConf.merge(config, config_new)
    return config


CONF = read_config()
PATH = Paths()
sparameters_path = PATH.sparameters


def rich_output() -> None:
    """Enables rich output."""
    try:
        from rich import pretty

        pretty.install()
    except ImportError:
        print("You can install `pip install gdsfactory[full]` for better visualization")


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


def print_config(key: Optional[str] = None) -> None:
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


def set_plot_options(
    show_ports: bool = True,
    show_subports: bool = False,
    label_aliases: bool = False,
    new_window: bool = False,
    blocking: bool = False,
    zoom_factor: float = 1.4,
) -> None:
    """Set plot options for matplotlib."""
    from gdsfactory.quickplotter import set_quickplot_options

    set_quickplot_options(
        show_ports=show_ports,
        show_subports=show_subports,
        label_aliases=label_aliases,
        new_window=new_window,
        blocking=blocking,
        zoom_factor=zoom_factor,
    )


if __name__ == "__main__":
    # print(PATH.sparameters)
    # print_config()
    # print_version()
    print_version_raw()
    # print_version_pdks()
    # write_tech("tech.json")

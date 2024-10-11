"""Gdsfactory configuration."""

from __future__ import annotations

import importlib
import pathlib
import sys
import tempfile
from enum import Enum, auto
from typing import TYPE_CHECKING

from kfactory.conf import config, get_affinity
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    pass

__version__ = "8.13.2"
PathType = str | pathlib.Path

home = pathlib.Path.home()
cwd = pathlib.Path.cwd()
module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent
home_path = pathlib.Path.home() / ".gdsfactory"
diff_path = repo_path / "gds_diff"
logpath = home_path / "log.log"

get_number_of_cores = get_affinity

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


CONF = config
CONF.difftest_ignore_label_differences = False
CONF.difftest_ignore_sliver_differences = False
CONF.difftest_ignore_cell_name_differences = True
CONF.bend_radius_error_type = ErrorType.ERROR
CONF.layer_error_path = (1000, 0)
CONF.connect_use_mirror = False
CONF.max_cellname_length = 32
CONF.cell_layout_cache = True
CONF.pdk = "generic"
CONF.port_types = [
    "optical",  # optical ports
    "electrical",  # electrical ports
    "placement",  # placement ports (excluded in netlist extraction)
    "vertical_te",  # for grating couplers with TE polarization
    "vertical_tm",  # for grating couplers with TM polarization
    "electrical_rf",  # electrical ports for RF (high frequency)
    "pad",  # for pads
    "pad_rf",  # for RF pads
    "bump",  # for bumps
    "edge_coupler",  # for edge couplers
]


class Paths:
    module = module_path
    repo = repo_path
    results_tidy3d = home / ".tidy3d"
    generic_tech = module / "generic_tech"
    klayout = generic_tech / "klayout"
    klayout_tech = klayout
    klayout_lyp = klayout_tech / "layers.lyp"
    klayout_lyt = klayout_tech / "tech.lyt"
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

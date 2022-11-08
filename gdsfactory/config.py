"""Gdsfactory loads a configuration from 3 files, high priority overwrites low.

priority:

1. A config.yml found in the current working directory (highest priority)
2. ~/.gdsfactory/config.yml specific for the machine
3. the yamlpath_default in gdsfactory.tech.yml (lowest priority)

`CONFIG` has all your computer specific paths that we do not care to store

You can access the config dictionary with `print_config`
"""

__version__ = "5.55.0"
import io
import json
import os
import pathlib
import subprocess
from pathlib import Path
from pprint import pprint
from typing import Any, Iterable, Optional, Union

import omegaconf
from loguru import logger
from omegaconf import OmegaConf

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
layer_path = module_path / "klayout" / "tech" / "layers.lyp"

MAX_NAME_LENGTH = 32

logger.info(f"Load {str(module_path)!r} {__version__}")
logger.add(sink=logpath)


default_config = io.StringIO(
    """
plotter: matplotlib
sparameters_path: ${oc.env:HOME}/.gdsfactory/sparameters/generic
"""
)


class Paths:
    module = module_path
    repo = repo_path
    results_tidy3d = home / ".tidy3d"
    klayout = module / "klayout"
    klayout_tech = klayout / "tech"
    klayout_lyp = klayout_tech / "layers.lyp"


def read_config(
    yamlpaths: Iterable[PathType] = (yamlpath_default, yamlpath_home, yamlpath_cwd),
) -> omegaconf.DictConfig:
    CONFIG = OmegaConf.load(default_config)
    for yamlpath in set(yamlpaths):
        yamlpath = pathlib.Path(yamlpath)
        if os.access(yamlpath, os.R_OK) and yamlpath.exists():
            logger.info(f"loading tech config from {yamlpath}")
            CONFIG_NEW = OmegaConf.load(yamlpath)
            CONFIG = OmegaConf.merge(CONFIG, CONFIG_NEW)
    return CONFIG


CONF = read_config()
PATH = Paths()


CONFIG = dict(
    config_path=yamlpath_cwd.absolute(),
    repo_path=repo_path,
    module_path=module_path,
    gdsdir=module_path / "tests" / "gds",
    masks_path=repo_path / "mask",
    home=home,
    cwd=cwd,
)

CONFIG["gdslib"] = repo_path / "gdslib"
CONFIG["gdsdiff"] = repo_path / "gdslib" / "gds"
CONFIG["modes"] = repo_path / "gdslib" / "modes"
CONFIG["sparameters"] = CONFIG["gdslib"] / "sp"
CONFIG["interconnect"] = CONFIG["gdslib"] / "interconnect"
CONFIG["samples_path"] = module_path / "samples"
CONFIG["netlists"] = module_path / "samples" / "netlists"
CONFIG["components_path"] = module_path / "components"
CONFIG["schemas"] = module_path / "tests" / "schemas"
CONFIG["schema_netlist"] = module_path / "tests" / "schemas" / "netlist.json"

sparameters_path = CONFIG["sparameters"]


def print_config(key: Optional[str] = None) -> None:
    """Prints a key for the config or all the keys."""
    if key:
        if CONFIG.get(key):
            print(CONFIG[key])
        else:
            print(f"`{key}` key not found in {CONFIG.keys()}")
    else:
        pprint(CONFIG)


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
    # print_config("gdslib")
    # print(CONFIG["git_hash"])
    # print(CONFIG["sparameters"])
    # print(CONFIG)
    print(CONF["sparameters_path"])
    # print_config()
    # write_tech("tech.json")

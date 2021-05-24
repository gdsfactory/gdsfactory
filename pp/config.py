"""gdsfactory loads a configuration from 3 files, high priority overwrites low priority:

1. A config.yml found in the current working directory (highest priority)
2. ~/.gdsfactory/config.yml specific for the machine
3. the yamlpath_default in pp/tech.yml (lowest priority)

`CONFIG` has all your computer specific paths that we do not care to store
`TECH` has all the useful info that we will store to have reproduceable layouts.

You can access all the config dictionary with `print_config` as well as a particular key

"""

__version__ = "2.5.0"
import json
import logging
import os
import pathlib
import subprocess
import tempfile
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Iterable, Optional, Union

from omegaconf import OmegaConf

from pp.tech import TECH

PathType = Union[str, pathlib.Path]

home = pathlib.Path.home()
cwd = pathlib.Path.cwd()
module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent
home_path = pathlib.Path.home() / ".gdsfactory"
diff_path = repo_path / "gds_diff"

yamlpath_cwd = cwd / "config.yml"
yamlpath_default = module_path / "config.yml"
yamlpath_home = home_path / "config.yml"
layer_path = module_path / "klayout" / "tech" / "layers.lyp"

dirpath_build = pathlib.Path(tempfile.TemporaryDirectory().name)
dirpath_test = pathlib.Path(tempfile.TemporaryDirectory().name)
MAX_NAME_LENGTH = 32


try:
    from git import InvalidGitRepositoryError, Repo

    try:
        git_hash = Repo(repo_path, search_parent_directories=True).head.object.hexsha
        git_hash_cwd = Repo(cwd, search_parent_directories=True).head.object.hexsha
    except InvalidGitRepositoryError:
        git_hash = None
        git_hash_cwd = None

except ImportError:
    pass


def read_config(
    yamlpaths: Iterable[PathType] = (yamlpath_default, yamlpath_home, yamlpath_cwd),
) -> Dict[str, Any]:
    CONFIG = OmegaConf.create()
    for yamlpath in set(yamlpaths):
        # print(f"loading tech config from {yamlpath}")
        if os.access(yamlpath, os.R_OK) and yamlpath.exists():
            CONFIG_NEW = OmegaConf.load(yamlpath)
            CONFIG = OmegaConf.merge(CONFIG, CONFIG_NEW)
    CONFIG.info = CONFIG.info or {}
    CONFIG.info.version = __version__
    return CONFIG


CONF = read_config()


def add_repo_information(TECH):
    TECH.info = TECH.info or {}
    TECH.info.git_hash = git_hash
    TECH.info.git_hash_cwd = git_hash_cwd


CONFIG = dict(
    config_path=yamlpath_cwd.absolute(),
    repo_path=repo_path,
    module_path=module_path,
    gdsdir=module_path / "gds",
    font_path=module_path / "gds" / "alphabet.gds",
    masks_path=repo_path / "mask",
    home=home,
    cwd=cwd,
)

mask_name = "notDefined"


if "mask" in CONF:
    mask_name = CONF.mask.name
    mask_config_directory = cwd
    build_directory = mask_config_directory / "build"
    CONFIG["devices_directory"] = mask_config_directory / "devices"
    CONFIG["mask_gds"] = mask_config_directory / "build" / "mask" / f"{mask_name}.gds"
else:
    dirpath_build.mkdir(exist_ok=True)
    build_directory = dirpath_build
    mask_config_directory = dirpath_build


# CONFIG["custom_components"] = TECH.custom_components
CONFIG["gdslib"] = repo_path / "gdslib"
CONFIG["sp"] = CONFIG["gdslib"] / "sp"
CONFIG["gds"] = CONFIG["gdslib"] / "gds"
CONFIG["gdslib_test"] = dirpath_test

CONFIG["build_directory"] = build_directory
CONFIG["gds_directory"] = build_directory / "devices"
CONFIG["cache_doe_directory"] = build_directory / "cache_doe"
CONFIG["doe_directory"] = build_directory / "doe"
CONFIG["mask_directory"] = build_directory / "mask"
CONFIG["mask_gds"] = build_directory / "mask" / (mask_name + ".gds")
CONFIG["mask_config_directory"] = mask_config_directory
CONFIG["samples_path"] = module_path / "samples"
CONFIG["netlists"] = module_path / "samples" / "netlists"
CONFIG["components_path"] = module_path / "components"

if "gds_resources" in CONFIG:
    CONFIG["gds_resources"] = CONFIG["masks_path"] / CONFIG["gds_resources"]

build_directory.mkdir(exist_ok=True)
CONFIG["gds_directory"].mkdir(exist_ok=True)
CONFIG["doe_directory"].mkdir(exist_ok=True)
CONFIG["mask_directory"].mkdir(exist_ok=True)
CONFIG["gdslib_test"].mkdir(exist_ok=True)


logging.basicConfig(
    filename=CONFIG["build_directory"] / "log.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)
logging.warning("This will get logged to a file")


def print_config(key: Optional[str] = None) -> None:
    """Prints a key for the config or all the keys"""
    if key:
        if CONFIG.get(key):
            print(CONFIG[key])
        else:
            print(f"`{key}` key not found in {CONFIG.keys()}")
    else:
        pprint(CONFIG)
        print(OmegaConf.to_yaml(TECH))


def complex_encoder(z):
    if isinstance(z, pathlib.Path):
        return str(z)
    else:
        type_name = type(z)
        raise TypeError(f"Object {z} of type {type_name} is not serializable")


def write_config(config: Any, json_out_path: Path) -> None:
    """Write config to a JSON file."""
    with open(json_out_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=complex_encoder)


def write_tech(json_out_path: Path) -> None:
    """Write config to a JSON file."""
    with open(json_out_path, "w") as f:
        json.dump(TECH, f, indent=2, sort_keys=True, default=complex_encoder)


def call_if_func(f: Any, **kwargs) -> Any:
    """Calls function if it's a function
    Useful to create objects from functions
    if it's an object it just returns the object
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
    print(TECH.layer.WG)
    # print(TECH)
    # print_config("gdslib")
    # print_config()
    # print(CONFIG["git_hash"])
    # print(CONFIG["sp"])
    # print(CONFIG)

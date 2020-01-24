""" gdsfactory loads a configuration from 3 files, high priority overwrites low priority:

1. A config.yml found in the current working directory (highest priority)
2. ~/.gdsfactory/config.yml specific for the machine
3. the default_config is in this file (lowest priority)

"""

__version__ = "1.1.1"
__all__ = ["CONFIG", "load_config", "write_config"]

import os
import json
import subprocess
import pathlib
import ast
import logging
from pprint import pprint

import hiyapyco
import numpy as np
from git import Repo


default_config = """
tech: generic
cache_url: 
BBOX_LAYER_EXCLUDE: "[]"
"""

home = pathlib.Path.home()
cwd = pathlib.Path.cwd()
roots = [pathlib.Path("/"), pathlib.Path("C:\\")]

module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent
home_path = pathlib.Path.home() / ".gdsfactory"
home_path.mkdir(exist_ok=True)

cwd_config = cwd / "config.yml"
home_config = home_path / "config.yml"

def load_config(cwd_config=cwd_config):
    """ loads config.yml and returns a dict with the config """
    cwd = cwd_config.parent
    # Find cwd config going up recursively
    while cwd not in roots:
        cwd_config = cwd / "config.yml"
        if os.path.exists(cwd_config):
            break
        cwd = cwd.parent
        
        if str(cwd).count("\\")<=1 and str(cwd).endswith("\\"):
            """
            Ensure the loop terminates on a windows machine
            """
            break

    CONFIG = hiyapyco.load(
        default_config,
        str(home_config),
        str(cwd_config),
        failonmissingfiles=False,
        loglevelmissingfiles=logging.DEBUG,
    )

    CONFIG["config_path"] = cwd_config.absolute()
    CONFIG["repo_path"] = repo_path
    CONFIG["module_path"] = module_path
    CONFIG["font_path"] = module_path / 'gds' / 'alphabet.gds'
    CONFIG["masks_path"] = repo_path / "mask"
    CONFIG["version"] = __version__
    CONFIG["home"] = home
    CONFIG["cwd"] = cwd

    if CONFIG.get("mask"):
        mask_name = CONFIG["mask"]["name"]
        mask_root_directory = cwd
        build_directory = mask_root_directory / "build"
        CONFIG["devices_directory"] = mask_root_directory / "devices"
        CONFIG["mask"]["gds"] = (
            mask_root_directory / "build" / "mask" / (mask_name + ".gds")
        )
    else:
        build_directory = home_path / "build"
        mask_root_directory = home_path / "build"

    if "custom_components" not in CONFIG:
        CONFIG["custom_components"] = None

    if "gdslib" not in CONFIG:
        CONFIG["gdslib"] = repo_path / "gdslib"
    CONFIG["gdslib_test"] = home_path / "gdslib_test"

    CONFIG["build_directory"] = build_directory
    CONFIG["log_directory"] = build_directory / "log"
    CONFIG["gds_directory"] = build_directory / "devices"
    CONFIG["cache_doe_directory"] = build_directory / "cache_doe"
    CONFIG["doe_directory"] = build_directory / "doe"
    CONFIG["mask_directory"] = build_directory / "mask"
    CONFIG["mask_root_directory"] = mask_root_directory
    CONFIG["gdspath"] = build_directory / "gds.gds"
    CONFIG["samples_path"] = repo_path / "samples"
    CONFIG["templates"] = repo_path / "templates"
    CONFIG["components_path"] = module_path / "components"

    if "gds_resources" in CONFIG:
        CONFIG["gds_resources"] = CONFIG["masks_path"] / CONFIG["gds_resources"]

    build_directory.mkdir(exist_ok=True)
    CONFIG["log_directory"].mkdir(exist_ok=True)
    CONFIG["gds_directory"].mkdir(exist_ok=True)
    CONFIG["doe_directory"].mkdir(exist_ok=True)
    CONFIG["mask_directory"].mkdir(exist_ok=True)
    CONFIG["gdslib_test"].mkdir(exist_ok=True)

    return CONFIG


def complex_encoder(z):
    if isinstance(z, pathlib.Path):
        return str(z)
    else:
        type_name = type(z)
        raise TypeError(f"Object {z} of type {type_name} is not serializable")


def write_config(config, json_out_path):
    with open(json_out_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=complex_encoder)


CONFIG = load_config()
CONFIG["grid_unit"] = 1e-6
CONFIG["grid_resolution"] = 1e-9
CONFIG["bend_radius"] = 10.0
CONFIG["layer_label"] = 66

try:
    CONFIG["git_hash"] = Repo(repo_path).head.object.hexsha
except Exception:
    CONFIG["git_hash"] = __version__


def print_config(key=None):
    if key:
        if CONFIG.get(key):
            print(CONFIG[key])
        else:
            print(f"`{key}` key not found in {cwd_config}")
    else:
        pprint(CONFIG)


def call_if_func(f, **kwargs):
    return f(**kwargs) if callable(f) else f


def parse_layer_exclude(l):
    return list(ast.literal_eval(l))


def get_git_hash():
    """ Get the current git hash """
    try:
        with open(os.devnull, "w") as shutup:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=shutup)
                .decode("utf-8")
                .strip("\n")
            )
    except subprocess.CalledProcessError:
        return "not_a_git_repo"


GRID_UNIT = CONFIG["grid_unit"]
GRID_RESOLUTION = CONFIG["grid_resolution"]

GRID_PER_UNIT = GRID_UNIT / GRID_RESOLUTION

GRID_ROUNDING_RESOLUTION = int(np.log10(GRID_PER_UNIT))
BEND_RADIUS = CONFIG["bend_radius"]

WG_EXPANDED_WIDTH = 2.5
TAPER_LENGTH = 35.0


CONFIG["BBOX_LAYER_EXCLUDE"] = parse_layer_exclude(CONFIG["BBOX_LAYER_EXCLUDE"])

if __name__ == "__main__":
    # print_config("gdslib")
    # print_config()
    print(CONFIG["git_hash"])

""" gdsfactory loads a configuration from 3 files, high priority overwrites low priority:

1. A config.yml found in the current working directory (highest priority)
2. ~/.gdsfactory/config.yml specific for the machine
3. the default_config is in this file (lowest priority)

"""

__version__ = "1.1.6"
__all__ = ["CONFIG", "write_config", "load_config"]

import os
import json
import subprocess
import pathlib
import ast
import logging
from pprint import pprint

from dotmap import DotMap
import hiyapyco
import numpy as np
from git import Repo


class ConfigMap(DotMap):
    def __getitem__(self, k):
        if (
            k not in self._map
            and self._dynamic
            and k != "_ipython_canary_method_should_not_exist_"
        ):
            # automatically extend to new DotMap
            self[k] = self.__class__()
            # print(f"{k} not in {sorted(list(self.keys()))}")
            # raise KeyError(f"{k} not in {sorted(list(self.keys()))}")
        return self._map[k]


default_config = """
tech: generic
BBOX_LAYER_EXCLUDE: "[]"
with_settings_label: True
layers:
    WG: [1, 0]
    WGCLAD: [1, 9]
    LABEL: [66, 0]
    label: [66, 0]
    WGCLAD: [111, 0]
    SLAB150: [2, 0]
    SLAB90: [3, 0]
    DEEPTRENCH: [7, 0]
    WGN: [34, 0]
    HEATER: [47, 0]
    M1: [41, 0]
    M2: [45, 0]
    M3: [49, 0]
    VIA1: [40, 0]
    VIA2: [44, 0]
    VIA3: [43, 0]
    NO_TILE_SI: [63, 0]
    PADDING: [68, 0]
    TEXT: [66, 0]
    PORT: [60, 0]
    label: [60, 0]
    LABEL: [201, 0]
    INFO_GEO_HASH: [202, 0]
    polarization_te: [203, 0]
    polarization_tm: [204, 0]

layer_colors:
    WG: ['gray', 1]

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


def load_config(path_config=cwd_config):
    cwd = path_config.parent

    # Find cwd config going up recursively
    while cwd not in roots:
        cwd_config = cwd / "config.yml"
        if cwd_config.exists():
            break
        cwd = cwd.parent

        if str(cwd).count("\\") <= 1 and str(cwd).endswith("\\"):
            """
            Ensure the loop terminates on a windows machine
            """
            break

    # print(cwd_config)
    CONFIG = hiyapyco.load(
        default_config,
        str(home_config),
        str(cwd_config),
        failonmissingfiles=False,
        loglevelmissingfiles=logging.DEBUG,
    )

    CONFIG["config_path"] = cwd_config or "notFound"
    CONFIG["repo_path"] = repo_path
    CONFIG["module_path"] = module_path
    CONFIG["font_path"] = module_path / "gds" / "alphabet.gds"
    CONFIG["masks_path"] = repo_path / "mask"
    CONFIG["version"] = __version__
    CONFIG["home"] = home
    CONFIG["cwd"] = cwd

    if CONFIG.get("mask"):
        mask_name = CONFIG["mask"]["name"]
        mask_config_directory = cwd
        build_directory = mask_config_directory / "build"
        CONFIG["devices_directory"] = mask_config_directory / "devices"
        CONFIG["mask"]["gds"] = (
            mask_config_directory / "build" / "mask" / (mask_name + ".gds")
        )
    else:
        build_directory = home_path / "build"
        mask_config_directory = home_path / "build"

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
    CONFIG["mask_config_directory"] = mask_config_directory
    CONFIG["gdspath"] = build_directory / "gds.gds"
    CONFIG["samples_path"] = module_path / "samples"
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


CONFIG = load_config()


def complex_encoder(z):
    if isinstance(z, pathlib.Path):
        return str(z)
    else:
        type_name = type(z)
        raise TypeError(f"Object {z} of type {type_name} is not serializable")


def write_config(config, json_out_path):
    with open(json_out_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=complex_encoder)


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


CONFIG["grid_unit"] = 1e-6
CONFIG["grid_resolution"] = 1e-9
CONFIG["bend_radius"] = 10.0

GRID_UNIT = CONFIG["grid_unit"]
GRID_RESOLUTION = CONFIG["grid_resolution"]
GRID_PER_UNIT = GRID_UNIT / GRID_RESOLUTION
GRID_ROUNDING_RESOLUTION = int(np.log10(GRID_PER_UNIT))
BEND_RADIUS = CONFIG["bend_radius"]
WG_EXPANDED_WIDTH = 2.5
TAPER_LENGTH = 35.0
CONFIG["BBOX_LAYER_EXCLUDE"] = parse_layer_exclude(CONFIG["BBOX_LAYER_EXCLUDE"])

layermap = CONFIG["layers"]
# LAYER = namedtuple("layermap", layer.keys())(*layer.values())
LAYER = ConfigMap(**layermap)

CONFIG.update(dict(cache_url=""))

try:
    CONFIG["git_hash"] = Repo(repo_path).head.object.hexsha
except Exception:
    CONFIG["git_hash"] = __version__


if __name__ == "__main__":
    # print_config("gdslib")
    print_config()
    # print(CONFIG["git_hash"])
    # print(CONFIG)

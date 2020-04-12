""" merges multiple JSONs:

"""

import json
import importlib
from git import Repo
from pp.config import logging, load_config, CONFIG, write_config, get_git_hash


def update_config_modules(config):
    """ update config with module git hashe and version (for each module in module_requirements section)
    """
    if config.get("module_requirements"):
        config.update({"git_hash": get_git_hash(), "module_versions": {}})
        for module_name in config["module_requirements"]:
            module = importlib.import_module(module_name)
            config["module_versions"].update(
                {
                    module_name: {
                        "version": module.__version__,
                        "git_hash": Repo(module.CONFIG["repo_path"]).head.object.hexsha,
                    }
                }
            )
    return config


def merge_json(config_path=CONFIG["cwd"] / "config.yml", json_version=6):
    """ Merge several JSON files from mask_config_directory
    requires a config.yml in the root of the mask directory

    Args:
        mask_config_directory: defaults to current working directory
        json_version: for maskhub parser
        jsons_filepaths: if we want to supply individual json files

    """
    logging.debug("Merging JSON files:")
    config = load_config(config_path)

    if config.get("mask") is None:
        raise ValueError(f"mask config missing from {config_path}")

    config = update_config_modules(config)

    mask_name = config["mask"]["name"]
    cell_directory = config["gds_directory"]
    json_out_path = config["mask_directory"] / (mask_name + ".json")
    doe_directory = config["build_directory"] / "doe"
    cache_doe_directory = config["cache_doe_directory"]

    cells = {
        d.stem: json.loads(open(d).read()) for d in cache_doe_directory.glob("*/*.json")
    }
    cells.update(
        {d.stem: json.loads(open(d).read()) for d in cell_directory.glob("*/*.json")}
    )

    does = {d.stem: json.loads(open(d).read()) for d in doe_directory.glob("*.json")}
    config.update(dict(json_version=json_version, cells=cells, does=does))

    write_config(config, json_out_path)
    print(f"Wrote  metadata in {json_out_path}")
    logging.info(f"Wrote  metadata in {json_out_path}")
    return config


if __name__ == "__main__":
    # from pprint import pprint

    config_path = CONFIG["samples_path"] / "mask_custom" / "config.yml"
    json_path = config_path.parent / "build" / "mask" / "sample_mask.json"
    d = merge_json(config_path)

    # print(config["module_versions"])
    # pprint(d['does'])

    # with open(json_path, "w") as f:
    #     f.write(json.dumps(d, indent=2))

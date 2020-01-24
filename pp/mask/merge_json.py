""" merges multiple JSONs:

"""

import json
import os
import importlib
from git import Repo
from pp.config import logging, load_config, CONFIG, write_config, get_git_hash


def update_config_modules(config):
    """ update config with module git hashe and version (for each module in module_requirements section)
    """
    if config.get('module_requirements'):
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


def merge_json(config=CONFIG, json_version=6):
    """ Merge several JSON files from mask_config_directory
    requires a config.yml in the root of the mask directory

    Args:
        mask_config_directory: defaults to current working directory
        json_version: for maskhub parser
        jsons_filepaths: if we want to supply individual json files
        
    """
    if config.get("mask") is None:
        raise ValueError(f"mask config missing from {config['cwd']}")

    config = update_config_modules(config)

    mask_name = config["mask"]["name"]
    jsons_directory = config["gds_directory"]
    json_out_path = config["mask_directory"] / (mask_name + ".json")

    cells = {}
    does = {}
    logging.debug("Merging JSON files:")

    for filename in jsons_directory.glob("*.json"):
        logging.debug(filename)
        with open(filename, "r") as f:
            data = json.load(f)
            if data.get("type") == "doe":
                does[data["name"]] = data
            else:
                cells.update(data.get("cells"))

    config.update({"json_version": json_version, "cells": cells, "does": does})
    write_config(config, json_out_path)
    logging.info("Wrote {}".format(os.path.relpath(json_out_path)))
    return config


if __name__ == "__main__":
    config_path = CONFIG["samples_path"] / "mask" / "config.yml"
    config = load_config(config_path)
    config = merge_json(config)
    # print(config["module_versions"])
    print(config)

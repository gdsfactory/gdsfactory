""" merges multiple JSONs:

"""

import importlib
import json

from git import Repo
from omegaconf import OmegaConf

from pp.config import CONFIG, conf, get_git_hash, logging, write_config


def update_config_modules(config=conf):
    """ update config with module git hashe and version (for each module in module_requirements section)
    """
    if config.get("requirements"):
        config.update({"git_hash": get_git_hash(), "module_versions": {}})
        for module_name in config["requirements"]:
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


def merge_json(
    doe_directory=CONFIG["doe_directory"],
    extra_directories=[CONFIG["gds_directory"]],
    jsonpath=CONFIG["mask_directory"] / "metadata.json",
    json_version=6,
    config=conf,
):
    """ Merge several JSON files from config.yml
    in the root of the mask directory, gets mask_name from there

    Args:
        mask_config_directory: defaults to current working directory
        json_version:

    """
    logging.debug("Merging JSON files:")
    cells = {}
    config = config or {}
    update_config_modules(config=config)

    for directory in extra_directories + [doe_directory]:
        for filename in directory.glob("*/*.json"):
            logging.debug(filename)
            with open(filename, "r") as f:
                data = json.load(f)
                cells.update(data.get("cells"))

    does = {d.stem: json.loads(open(d).read()) for d in doe_directory.glob("*.json")}
    metadata = dict(
        json_version=json_version,
        cells=cells,
        does=does,
        config=OmegaConf.to_container(config),
    )

    write_config(metadata, jsonpath)
    print(f"Wrote  metadata in {jsonpath}")
    logging.info(f"Wrote  metadata in {jsonpath}")
    return metadata


if __name__ == "__main__":
    d = merge_json()
    print(d)

    # print(config["module_versions"])
    # pprint(d['does'])

    # with open(jsonpath, "w") as f:
    #     f.write(json.dumps(d, indent=2))

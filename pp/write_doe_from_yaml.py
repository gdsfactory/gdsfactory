""" write DOE from YAML file """

import sys
import importlib

from pp.config import CONFIG
from pp.doe import load_does
from pp.write_doe import write_doe


def import_custom_doe_factories():
    """ Find if we have custom DOEs on this config.
    Make them available in component_type2factory
    """

    sys.path += [CONFIG["mask_root_directory"]]
    if CONFIG["custom_components"]:
        try:
            importlib.import_module(CONFIG["custom_components"])
        except Exception:
            pass


def write_doe_from_yaml(filepath):
    """ Loads DOE settings from yaml file and writes GDS into build_directory

    Args:
        filepath: YAML file describing DOE

    Returns:
        gdspaths: list

    For each DOE save:

    - GDS
    - json metadata
    - ports CSV
    - markdown report, with DOE settings
    """
    does = load_does(filepath)

    gdspaths = []
    for doe_name, doe in does.items():
        # print(doe_name)
        # print(doe.get("settings"))
        # print(doe.get("do_permutations"))
        # print(doe)
        # print(list(doe.keys()))
        # print(type(doe.get('settings')))
        # assert type(doe.get('settings'))
        d = write_doe(
            component_type=doe.get("component"),
            doe_name=doe_name,
            do_permutations=doe.get("do_permutations", True),
            list_settings=doe.get("settings"),
            description=doe.get("description"),
            analysis=doe.get("analysis"),
            test=doe.get("test"),
            functions=doe.get("functions"),
        )
        gdspaths.append(d)
    return gdspaths


def test_write_doe_from_yaml():
    does_path = CONFIG["samples_path"] / "mask" / "does.yml"
    gdspaths = write_doe_from_yaml(does_path)
    print(gdspaths)


if __name__ == "__main__":
    test_write_doe_from_yaml()

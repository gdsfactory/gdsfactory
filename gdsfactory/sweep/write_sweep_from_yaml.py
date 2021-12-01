"""Read Sweep parameters from YAML file and writes gdsfiles"""

import importlib
import io
import sys
from pathlib import Path
from typing import List, Union

from gdsfactory.config import CONFIG
from gdsfactory.sweep.read_sweep import read_sweep
from gdsfactory.sweep.write_sweep import write_sweep


def import_custom_doe_factories():
    """Find if we have custom DOEs on this config.
    Make them available in component_factory
    """

    sys.path += [CONFIG["mask_root_directory"]]
    if CONFIG["custom_components"]:
        try:
            importlib.import_module(CONFIG["custom_components"])
        except Exception:
            pass


def write_sweep_from_yaml(yaml: Union[str, Path]) -> List[List[Path]]:
    """Read DOE settings from yaml and writes GDS files build_directory

    Args:
        filepath: YAML string or filepath describing DOEs

    Returns:
        gdspaths: list

    For each DOE save:

    - GDS
    - json metadata
    - ports CSV
    - markdown report, with DOE settings
    """
    yaml = io.StringIO(yaml) if isinstance(yaml, str) and "\n" in yaml else yaml
    does = read_sweep(yaml)

    gdspaths = []
    for doe_name, doe in does.items():
        # print(doe_name)
        # print(sweep.get("settings"))
        # print(sweep.get("do_permutations"))
        # print(sweep)
        # print(list(sweep.keys()))
        # print(type(sweep.get('settings')))
        # assert type(sweep.get('settings'))
        d = write_sweep(
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


def test_write_doe_from_yaml() -> None:
    does_path = CONFIG["samples_path"] / "mask" / "does.yml"
    gdspaths = write_sweep_from_yaml(does_path)
    # print(len(gdspaths))
    assert len(gdspaths) == 4  # 2 does
    assert len(gdspaths[0]) == 2  # 2 GDS in the first DOE
    assert len(gdspaths[1]) == 2  # 2 GDS in the 2nd DOE


sample_yaml = """
mmi_width:
  component: mmi1x2
  settings:
    width_mmi: [4.5, 5.6]
    length_mmi: 10

mmi_width_length:
  component: mmi1x2
  do_permutation: True
  settings:
    length_mmi: [11, 12]
    width_mmi: [3.6, 7.8]

"""


def test_write_doe_from_yaml_string() -> None:
    gdspaths = write_sweep_from_yaml(sample_yaml)
    # print(len(gdspaths))
    assert len(gdspaths) == 2  # 2 does
    assert len(gdspaths[0]) == 2  # 2 GDS in the first DOE
    assert len(gdspaths[1]) == 4  # 4 GDS in the 2nd DOE


if __name__ == "__main__":
    test_write_doe_from_yaml()
    # test_write_doe_from_yaml_string()
    # gdspaths = write_doe_from_yaml(sample_yaml)
    # print(len(gdspaths))
    # print(len(gdspaths[0]))
    # print(len(gdspaths[1]))

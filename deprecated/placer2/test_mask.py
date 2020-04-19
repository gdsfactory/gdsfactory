import pathlib
import pp
from pp.config import load_config

from pp.placer import component_grid_from_yaml


def test_mask():
    workspace_folder = pathlib.Path(__file__).parent
    filepath_yml = workspace_folder / "config.yml"
    config = load_config(filepath_yml)
    gdspath = str(config["mask"]["gds"])

    top_level = component_grid_from_yaml(config)
    pp.write_gds(top_level, gdspath)
    assert config["mask"]["gds"].exists()
    return gdspath


if __name__ == "__main__":
    c = test_mask()
    pp.show(c)

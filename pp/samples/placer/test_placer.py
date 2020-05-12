import pathlib
import pp
from pp.config import load_config

from pp.placer import component_grid_from_yaml


def test_placer():
    cwd = pathlib.Path(__file__).parent
    filepath = cwd / "config.yml"
    config = load_config(filepath)
    gdspath = str(config["mask"]["gds"])

    top_level = component_grid_from_yaml(filepath)
    pp.write_gds(top_level, gdspath)
    assert config["mask"]["gds"].exists()
    return gdspath


if __name__ == "__main__":
    c = test_placer()
    pp.show(c)
